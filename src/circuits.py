from typing_extensions import Self, Any, Literal #For annotations only
import math
import src.circ_utils as circ_utils
from collections import deque, defaultdict
import numpy as np
import queue
import functools
import copy

import src.graph_mapping as graph_mapping
import networkx as nx

wire_t = circ_utils.wire_t

tsp_t = Literal['christofides']|Literal['greedy_tsp']|Literal['simulated_annealing_tsp']|Literal['threshold_accepting_tsp']
pmo_t = Literal['naive']|Literal['qubit_count']|Literal['depth_sum']

#class for graph nodes. Used to represent single gates and phase polynomials
#notes for use in comparisons: 
# * Node.nodeType compares types of nodes, ignoring parameters and additional data
# * Node.is_equivalent_to() compares if 2 nodes have equivalent data, including or excluding ins/outs based on a given parameter
# * == operator behavior is python default

class Node:
    #note that ins and outs need to be initialized before proper use, see connect_to() and connect_from()
    sig = 0 #<--- debug purposes only
    
    def __init__(this, wires:list[wire_t], ins:(dict[wire_t,Self]|None) = None, outs:(dict[wire_t,Self]|None) = None):
        this.wires = wires.copy()
        this.nodeType = "generic"
        this.sig = Node.sig
        this.latency = 1.0
        Node.sig += 1
        if ins != None:
            this.ins = ins.copy()
        else:
            this.ins:dict[int,Node] = dict()
        
        if outs != None:
            this.outs = outs.copy()
        else:
            this.outs:dict[int,Node] = dict()

    #makes edge directing this node to 1 other node
    def connect_to(this, target_node:Self, wire:wire_t):
        if this.outs.get(wire) != None:
            this.outs[wire].ins.pop(wire)
        if target_node == None:
            return
        if target_node.ins.get(wire) != None:
            target_node.ins[wire].outs.pop(wire)
        this.outs[wire] = target_node
        target_node.ins[wire] = this

    def connect_from(this, source_node:Self, wire:wire_t):
        if this.ins.get(wire) != None:
            this.ins[wire].outs.pop(wire)
        if source_node == None:
            return
        if source_node.outs.get(wire) != None:
            source_node.outs[wire].ins.pop(wire)
        this.ins[wire] = source_node
        source_node.outs[wire] = this

    #find previous node applying to a wire
    def prev(this, wire:wire_t):
        return this.ins.get(wire)
    
    #find next node applying to a wire
    def next(this, wire:wire_t):
        return this.outs.get(wire)
    
    #attach this node after a set of previous nodes, while remaking the graph edges correctly
    def attach_after(this, prev_nodes:dict[wire_t,Self]):
        for wire, node in prev_nodes.items():
            this.connect_to(node.outs[wire], wire)
            this.connect_from(node, wire)

    #attach this node before a set of next nodes, while remaking the graph edges correctly
    def attach_before(this, next_nodes:dict[wire_t,Self]):
        for wire, node in next_nodes.items():
            this.connect_from(node.ins[wire], wire)
            this.connect_to(node, wire)

    #remove this node from its current circuit, while remaking the graph edges correctly
    def remove_from_circuit(this):
        for q in set(this.ins.keys()) & set(this.outs.keys()):
            if this.prev(q) == None or this.next(q) == None:
                continue
            this.prev(q).connect_to(this.next(q), q)
        for q, node in this.ins.items():
            del node.outs[q]
        for q, node in this.outs.items():
            del node.ins[q]
        this.ins = dict()
        this.outs = dict()

    def shift_forward(this, wire:wire_t):
        next = this.next(wire)
        nnext = next.next(wire)
        prev = this.prev(wire)
        
        next.connect_to(this, wire)
        next.connect_from(prev, wire)
        this.connect_to(nnext, wire)

    def shift_backward(this, wire:wire_t):
        prev = this.prev(wire)
        pprev = prev.prev(wire)
        next = this.next(wire)
        
        prev.connect_to(next, wire)
        this.connect_from(pprev, wire)
        this.connect_to(prev, wire)

    def get_latecy(this):
        if len(this.wires) > 1:
            return 20
        return 1
        #return this.latency

    def to_circuit(this):
        return Circuit(ins=this.ins, outs=this.outs)

    def to_instr(this):
        return "<?>"
    
    def __repr__(this) -> str:
        return this.to_instr() + f" #{this.sig}"
    
    #checks if this and the other gate is equivalent, ignoring input/outputs nodes. 
    #should not be used to replace ==/__eq__, as that should be reserved for default object compare behavior
    def is_equivalent_to(this, other_node:Self, ignore_wires:bool = False):
        if this.__class__ != other_node.__class__:
            return False
        return this.nodeType == other_node.nodeType and (ignore_wires or this.wires == other_node.wires)
    
    def copy_disconnected(this):
        return Node(this.wires)
    
#subclass that represents gates
class Gate(Node):
    def __init__(this, gateType:str, wires:list[wire_t], params:list[float] = []):
        super().__init__(wires)
        this.nodeType = gateType
        this.params = params.copy()

    def to_instr(this):
        out = this.nodeType
        if this.params != None and len(this.params) > 0:
            out += f"({', '.join([str(p).strip('()') for p in this.params])})"

        wire_strs = []
        wire_strs.extend([f"q[{q}]" for t,q in this.wires if t == 'q'])
        wire_strs.extend([f"qm[{q}]" for t,q in this.wires if t == 'qm'])

        if len(wire_strs) > 0:
            out += " "+", ".join(wire_strs)
        return out + ';'
    
    def copy_disconnected(this):
        return Gate(this.nodeType, this.wires, this.params)
    
    def is_equivalent_to(this, other_node:Self, ignore_wires:bool = False):
        if not super().is_equivalent_to(other_node, ignore_wires=ignore_wires):
            return False
        return this.params == other_node.params

class CommentLine(Node):
    def __init__(this, wires:list[wire_t], message:str):
        super().__init__(wires)
        this.message = message
        this.nodeType = "commentLine"

    def to_instr(this):
        return '//'+this.message
    
    def copy_disconnected(this):
        return CommentLine(this.wires, this.message)
    
    def is_equivalent_to(this, other_node:Self, ignore_wires:bool = False):
        if not super().is_equivalent_to(other_node, ignore_wires=ignore_wires):
            return False
        return this.message == other_node.message
    
class PauliNode(Node):
    def __init__(this, pauli_string:str, displacement:float, wires:list[wire_t]|None = None):
        if wires == None:
            q_nums = [i for i,q in enumerate(pauli_string) if q != 'I']
            wires = []
            for q in q_nums:
                wires += [('q',q)]
        super().__init__(wires)
        this.pauli_string = pauli_string
        this.displacement = displacement
        this.nodeType = "pauliNode"

    def map_to_coupling_graph(this, mapping:graph_mapping.Mapping, tsp_method:tsp_t = 'christofides'):
        #print(this)
        all_phys_qubits = [ mapping.get_physical_from_log(q) for q in this.wires if q[0]=='q' and q[1] < len(this.pauli_string) and this.pauli_string[q[1]] != 'I']
        phys_qumodes = [('qm',q[1]) for q in all_phys_qubits]

        model_graph = mapping.coupling_graph.copy()

        for qm in phys_qumodes:
            model_graph.add_edge(qm, 'extra', weight=9999999)
        if tsp_method == 'greedy_tsp':
            tsp_method_f = nx.approximation.traveling_salesman.greedy_tsp
        elif tsp_method == 'simulated_annealing_tsp':
            tsp_method_f = lambda x, **kwargs : nx.approximation.traveling_salesman.simulated_annealing_tsp(x, "greedy", **kwargs)
        elif tsp_method == 'threshold_accepting_tsp':
            tsp_method_f = lambda x, **kwargs : nx.approximation.traveling_salesman.threshold_accepting_tsp(x, "greedy", **kwargs)
        else:
            tsp_method_f = nx.approximation.traveling_salesman.christofides
        path:list[wire_t] = nx.approximation.traveling_salesman_problem(model_graph, weight='weight', nodes=phys_qumodes+['extra'], cycle=False, method=tsp_method_f)
        #print(path)
        
        path.remove('extra')

        checkpoints = [path.index(qm) for qm in phys_qumodes]

        
        out = Circuit(list(sorted(set(path + all_phys_qubits))))
        for log,phys in [(q,mapping.get_physical_from_log(q)) for q in this.wires if q[0]=='q' and q[1] < len(this.pauli_string) and this.pauli_string[q[1]] != 'I']:
            pauli_gate = this.pauli_string[log[1]]
            if pauli_gate == 'Y':
                out.append_node(Gate("sdg", [('q',phys[1])]))
            if pauli_gate in ('X','Y'):
                out.append_node(Gate("h", [('q',phys[1])]))   
            
        #TODO put work here
        ancilla = path[0]
        def make_parity_path(is_reversed:bool = False):
            nonlocal ancilla, out
            ite = enumerate(path) 
            if is_reversed:
                ite = reversed(list(ite))    
            prev = None
            for i,qm in ite:
                if ancilla != qm:
                    out.append_node(Gate('BS', [qm, ancilla], [circ_utils.PiAngle(1), 0]))
                    out.append_node(Gate('R', [qm], [circ_utils.PiAngle(0.5)]))
                    out.append_node(Gate('R', [ancilla], [circ_utils.PiAngle(0.5)]))
                    ancilla = qm
                if i in checkpoints:
                    out.append_node(Gate('CP', [('q',qm[1]), ancilla]))
                prev = qm

        make_parity_path()
        out.append_node(Gate("D", [ancilla], [this.displacement]))
        make_parity_path(is_reversed=True)
        out.append_node(Gate("D", [ancilla], [this.displacement]))
        make_parity_path()
        out.append_node(Gate("D", [ancilla], [this.displacement]))
        make_parity_path(is_reversed=True)
        out.append_node(Gate("D", [ancilla], [this.displacement]))

        for log,phys in [(q,mapping.get_physical_from_log(q)) for q in this.wires if q[0]=='q' and q[1] < len(this.pauli_string) and this.pauli_string[q[1]] != 'I']:
            pauli_gate = this.pauli_string[log[1]]
            if pauli_gate in ('X','Y'):
                out.append_node(Gate("h", [('q',phys[1])]))
                if pauli_gate == 'Y':
                    out.append_node(Gate("s", [('q',phys[1])]))
  

        return out

    def to_instr(this):
        return f'pauli({this.displacement}) {this.pauli_string};'
    
    def copy_disconnected(this):
        return PauliNode(this.pauli_string, this.displacement, this.wires)
    
    def is_equivalent_to(this, other_node:Self, ignore_wires:bool = False):
        if not super().is_equivalent_to(other_node, ignore_wires=ignore_wires):
            return False
        return this.pauli_string == other_node.pauli_string and this.displacement == other_node.displacement

#actual circuit class. Should work as a graph.
class Circuit:
    # For input params, its either (size:int) or (wires:list[int]) for a new circuit, or (ins:dict[int,Node], outs,dict[int,Node]) for a subcircuit. 
    
    # Note: It is suggested against making multiple subcircuits on the same circuit due to how the ins/outs system works.
    # Any alternate solutions to define circuits in ways other than exclusive ins/outs are welcome, but remember that this should still support empty circuits.
    def __init__(this, wires:(list[wire_t]|None) = None, qubit_size:int|None = None, qumode_size:int|None = None, ins:(dict[wire_t,Node]|None) = None, outs:(dict[wire_t,Node]|None) = None):
        if ins != None and outs != None:
            if set(ins.keys()) != set(outs.keys()):
                raise Exception("Invalid")
            this.wires = sorted(list(ins.keys()))
            this.ins = ins.copy()
            this.outs = outs.copy()
            this.is_subcircuit = True #temp measure
            #this.get_sequence()
            return
        if qubit_size == None and qumode_size == None and wires == None:
            raise Exception("Invalid")
        if wires != None:
            this.wires = wires
        else:
            this.wires = []
            if qubit_size != None:
                this.wires.extend([('q',i) for i in range(qubit_size)])
            if qumode_size != None:
                this.wires.extend([('qm',i) for i in range(qubit_size)])
        
        this.ins:dict[wire_t,Node] = dict()
        this.outs:dict[wire_t,Node] = dict()
        this.is_subcircuit = False
        for q in this.wires:
            this.ins[q] = Node([q])
            this.outs[q] = Node([q])
            this.ins[q].connect_to(this.outs[q], q)

    #returns all nodes in the circuit as a list of nodes, ordered with dependency in mind.
    def get_sequence(this):
        sequence:list[Node] = []
        tracked = set()
        buffer:deque[Node] = deque()
        for q, in_node in this.ins.items():
            tracked.add(in_node)
            node = in_node.next(q)
            if in_node not in node.ins.values():
                raise Exception(f"Improper Graph Structure: {this.wires}, {this.ins} -> {this.outs}")
            if node in this.ins.values():
                raise Exception(f"Improper Graph Structure: {this.wires}, {this.ins} -> {this.outs}")
            if node not in buffer and node not in tracked:
                buffer.append(node)

        skip_streak = 0
        #print("\nget_seq buffer:")
        while len(buffer) > skip_streak:
            #print(buffer)
            target = buffer.popleft()
            if not all(prev_node in tracked for prev_node in target.ins.values()) and target not in this.outs.values():
                skip_streak += 1
                buffer.append(target)
                continue
            skip_streak = 0
            if target not in this.outs.values():
                for node in target.outs.values():
                    if target not in node.ins.values():
                        raise Exception(f"Improper Graph Structure: {this.wires}, {this.ins} -> {this.outs}")
                    if node not in buffer and node not in tracked:
                        buffer.append(node)
                sequence.append(target)
            tracked.add(target)

        if len(buffer) != 0:
            raise Exception(f"Improper Graph Structure: {this.wires}, {this.ins} -> {this.outs}\n buffer:{buffer}")
        return sequence

    def append_node(this, new_node:Node, force:bool = False):
        if any(q not in this.wires for q in new_node.wires):
            if force:
                for q in new_node.wires:
                    if q not in this.wires:
                        this.extend_wires(q)
            else:
                raise Exception(f"Invalid wire input: {[q for q in new_node.wires if q not in this.wires]}")
        
        prev_nodes:dict[int,Node] = dict()
        for q in new_node.wires:
            prev_nodes[q] = this.outs[q].prev(q)
        new_node.attach_after(prev_nodes)

    #TODO: Untested
    def append_circuit(this, new_circuit:Self):
        if any(q not in this.wires for q in new_circuit.wires):
            raise Exception("Invalid wire input")
        
        for q,n in new_circuit.ins.items():
            this.outs[q].prev(q).connect_to(n.next(q), q)
        for q,n in new_circuit.outs.items():
            n.prev(q).connect_to(this.outs[q], q)

    
    #adds gate to end of circuit
    #deprecated in favor of append_node(Gate(...)) 
    '''
    def append_new_gate(this, gateType:str, wires:list[int] = None, params:list[float] = None):
        if any(q not in this.wires for q in wires):
            raise Exception("Invalid wire input")
        
        new_gate:Gate = Gate(gateType, wires, params)
        prev_nodes:dict[int,Node] = dict()
        for q in wires:
            prev_nodes[q] = this.outs[q].prev(q)
        new_gate.attach_after(prev_nodes)
    '''
    #remaps the circuits wires (very jank)
    #In particlur, the function input parameter is very awkward to use. See remap_demo.py
    def remap(this, mapping):
        if this.is_subcircuit:
            raise Exception("Illegal Function") #yes, this is horrible design. I'll refactor this eventually.
        
        def remap_list(in_list:list[int]):
            out_list:list[int] = []
            for num in in_list: 
                out_list.append(mapping(num))
            return out_list
        
        def remap_dict(in_dict:dict[int]):
            out_dict:dict[int] = dict()
            for i, node in in_dict.items():
                new_i = mapping(i)
                if new_i in out_dict.keys():
                    raise Exception("Mapping error: wire Overlap")
                out_dict[new_i] = node
            return out_dict

        this.wires = remap_list(this.wires)

        for in_node in this.ins.values():
            in_node.wires = remap_list(in_node.wires)
            in_node.outs = remap_dict(in_node.outs)
        this.ins = remap_dict(this.ins)
        
        sequence = this.get_sequence()
        for node in sequence:
            node.wires = remap_list(node.wires)
            node.ins = remap_dict(node.ins)
            node.outs = remap_dict(node.outs)
        
        for out_node in this.outs.values():
            out_node.wires = remap_list(out_node.wires)
            out_node.ins = remap_dict(out_node.ins)
        this.outs = remap_dict(this.outs)
        
        this.get_sequence() # this is to verify the circuit structure, for debug purposes

    #creates a copy, with different node objects
    def copy(this):
        out_circuit = Circuit(wires=this.wires)
        for node in this.get_sequence():
            out_circuit.append_node(node.copy_disconnected())
        return out_circuit
    
    #copy the target circuit to the given inputs/outputs. 
    #Note that the parameter circuit is the circuit to be replaced and must match the given circuit's wires.
    #the circuit object the function is called at will be unchanged.
    #Below function is completely untested as of note, use at your own risk (for now)
    def replace_at(this, subcircuit_to_replace:Self):
        if not (set(this.wires) == set(subcircuit_to_replace.wires)):
            raise Exception("Invalid wires for Parameters")
        sequence = this.get_sequence()
        for node in sequence:
            if not set(node.wires).issubset(set(this.wires)):
                raise Exception("Circuit to be placed is not independent")
        
        for q in subcircuit_to_replace.wires:
            subcircuit_to_replace.ins[q].connect_to(subcircuit_to_replace.outs[q], q)
        for node in sequence:
            if not set(node.wires).issubset(set(subcircuit_to_replace.wires)):
                raise Exception("Circuit to replace is not independent")
            subcircuit_to_replace.append_node(node.copy_disconnected())

    #TODO
    def find_match(this, inputs:dict[wire_t,Node]):
        subcirc:Circuit; mapping:list[tuple[int,int]]
        return subcirc, mapping

    #Checks to see if this and another circuit/subcircuit match gates. remap is an optional function to remap other_circuit
    def structural_match(this, other_circuit:Self, remap = None):
        if remap == None:
            remap = lambda x : x
        for q in this.wires:
            c1ptr = this.ins[q].next(q)
            if remap(q) not in other_circuit.ins.keys():
                return False
            c2ptr = other_circuit.ins[remap(q)].next(remap(q))
            while True:
                if c1ptr == this.outs[q] and c2ptr == other_circuit.outs[remap(q)]:
                    break
                if c1ptr != this.outs[q] and c2ptr != other_circuit.outs[remap(q)]:
                    if not c1ptr.is_equivalent_to(c2ptr):
                        return False
                else:
                    return False
                c1ptr = c1ptr.next(q)
                c2ptr = c2ptr.next(remap(q))
        return True

    def depths(this):
        current_depths:defaultdict[wire_t, float] = defaultdict(float)
        for n in this.get_sequence():
            current_indices = n.wires
            depth = max(current_depths[i] for i in current_indices) + n.get_latecy()
            for i in current_indices:
                current_depths[i] = depth

        return current_depths
    
    def extend_wires(this, new_wire:wire_t):
        if this.is_subcircuit:
            raise Exception("Subcircuits should not be extended with extend_wires")
        this.ins[new_wire] = Node([new_wire])
        this.outs[new_wire] = Node([new_wire])
        this.ins[new_wire].connect_to(this.outs[new_wire], new_wire)
        this.wires.append(new_wire)

    def map_to_graph(this, c_graph:nx.Graph, pauli_mapping_order:pmo_t = 'naive', tsp_method:tsp_t = 'christofides', qubit_swaps:bool = True):     
        if len(this.wires) > len(c_graph.nodes):
            raise Exception("hardware graph too small")
        current_mapping = graph_mapping.Mapping(c_graph)

        #currently just index mapping
        for w in this.wires:
            current_mapping.map_qubit(w, w)

        out = Circuit(list(c_graph.nodes))
        tracked = set()
        buffer:list[Node] = []
        for q, in_node in this.ins.items():
            tracked.add(in_node)
            node:Node = in_node.next(q)
            if in_node not in node.ins.values():
                raise Exception(f"Improper Graph Structure: {this.wires}, {this.ins} -> {this.outs}")
            if node in this.ins.values():
                raise Exception(f"Improper Graph Structure: {this.wires}, {this.ins} -> {this.outs}")
            if node not in buffer and node not in tracked:
                buffer.append(node)
            if node.nodeType != 'pauliNode':
                continue
            ptr = node.next(q)
            while ptr != None and ptr not in this.outs and ptr.nodeType == 'pauliNode':
                if ptr not in buffer and ptr not in tracked:
                    buffer.append(ptr)
                ptr = ptr.next(q)

        depths:defaultdict[float] = defaultdict(float)
        def update_depth(node:Node|list[Node], multiple:bool = False):
            if multiple:
                for n in node:
                    update_depth(n)
                return
            current_indices = node.wires
            depth = max(depths[i] for i in current_indices) + node.get_latecy()
            for i in current_indices:
                depths[i] = depth

        #print(f"starting buff {buffer}")
        
        #print("\nget_seq buffer:")
        viable_targets:list[Node] = []
        while True:

            #print("loop")

            #get target
            def is_viable(node:Node):
                for w,n in node.ins.items():
                    prev_node = n
                    while prev_node.nodeType == 'pauliNode' and prev_node not in this.ins.values():
                        if node.nodeType != 'pauliNode':
                            if prev_node not in tracked:
                                return False
                        prev_node = prev_node.prev(w)
                    if prev_node not in tracked:
                        return False
                return True
            remov_set = set()
            for n in buffer:
                if is_viable(n) and n not in viable_targets:
                    viable_targets.append(n)
                    remov_set.add(n)
            for n in remov_set:
                buffer.remove(n)
            buffer = [n for n in buffer if n not in viable_targets]
            #print("buffer", buffer)
            #print("viable_targets", viable_targets)
            if len(viable_targets) <= 0:
                break

            #Select from viable targets
            viable_paulis = [n for n in viable_targets if n.nodeType == 'pauliNode']
            if len(viable_paulis) > 0:
                #inser other algorithms here later
                if pauli_mapping_order == 'qubit_count':
                    target:PauliNode = min(viable_paulis, key=lambda x : len(x.wires))
                elif pauli_mapping_order == 'depth_sum':
                    target:PauliNode = min(viable_paulis, key=lambda x : len(x.wires)+sum([depths[w] for w in x.wires]))
                else:
                    target:PauliNode = viable_paulis[0]
                viable_paulis.remove(target)
                viable_targets.remove(target)
            else:
                #print(f"select from {viable_targets}")
                flag = True
                curr_cost = None
                mappings_curr_cost = set()
                while flag:

                    for q in viable_targets:
                        if len(q.wires) < 2 or nx.is_connected(c_graph.subgraph([current_mapping.get_physical_from_log(w) for w in q.wires])):
                            target = q
                            viable_targets.remove(q)
                            flag = False
                            break
                    else:
                        #include swap mapping here
                        swaps:set[tuple] = set()
                        for n in viable_targets:
                            for q in n.wires:
                                if q[0] != 'qm':
                                    continue
                                phys_q = current_mapping.get_physical_from_log(q)
                                for q2 in current_mapping.coupling_graph.neighbors(phys_q):
                                    if q2[0] != 'qm':
                                        continue
                                    swap = tuple(sorted([phys_q, q2]))
                                    if swap not in swaps:
                                        swaps.add(swap)
                        def swap_cost(swap:tuple|None):

                            cost_tracker = dict()

                            min_cost = math.inf
                            total_cost = 0
                            new_mapping = current_mapping.copy()
                            if swap != None:
                                new_mapping.swap_physical_qubits(swap[0], swap[1])
                            for n in viable_targets:
                                cost = 0
                                for q1 in n.wires:
                                    for q2 in n.wires:
                                        if q1 <= q2:
                                            continue
                                        short_path_len = nx.shortest_path_length(new_mapping.coupling_graph, new_mapping.get_physical_from_log(q1), new_mapping.get_physical_from_log(q2))
                                        if q1[0] == 'q' or q2[0] == 'q':
                                            short_path_len *= 2
                                        cost += short_path_len
                                if cost < min_cost:
                                    min_cost = cost
                                total_cost += cost
                            return (min_cost, total_cost)
                        #print(f"swap costs: {list(sorted([(s,swap_cost(s)) for s in swaps]))}")
                        target_swap:tuple = min(swaps, key=swap_cost)

                        bs_gate = Gate('BS', [target_swap[0], target_swap[1]], [circ_utils.PiAngle(1), 0])
                        r1_gate = Gate('R', [target_swap[0]], [circ_utils.PiAngle(0.5)])
                        r2_gate = Gate('R', [target_swap[1]], [circ_utils.PiAngle(0.5)])
                        out.append_node(bs_gate)
                        update_depth(bs_gate)
                        out.append_node(r1_gate)
                        update_depth(r1_gate)
                        out.append_node(r2_gate)
                        update_depth(r2_gate)
                        #print(f"swap {target_swap[0]} <-> {target_swap[1]} ({swap_cost(None)} -> {swap_cost(target_swap)})")
                        #print(current_mapping)
                        #print(viable_targets)
                        current_mapping.swap_physical_qubits(target_swap[0],target_swap[1])
                        continue

            if target not in this.outs.values():
                for node in target.outs.values():
                    if target not in node.ins.values():
                        raise Exception(f"Improper Graph Structure: {this.wires}, {this.ins} -> {this.outs}")
                    if node in this.outs.values():
                        continue
                    if node not in buffer and node not in tracked:
                        buffer.append(node)
                    if target.nodeType == 'pauliNode':
                        ptr = node.next(q)
                        while ptr != None and ptr not in this.outs and ptr.nodeType == 'pauliNode':
                            if ptr not in buffer and ptr not in tracked:
                                buffer.append(ptr)
                            ptr = ptr.next(q)

                #apply node
                #print("mapping:", target)
                if target.nodeType == 'pauliNode':
                    target:PauliNode

                    #something something qubit swaps
                    if qubit_swaps:
                        
                        def is_connected(list_qubits:list[wire_t], graph:nx.Graph):
                            phys_qumodes = [('qm', q[1]) for q in list_qubits]
                            return nx.is_connected(graph.subgraph(phys_qumodes))
                        def get_avg_dist(list_qubits:list[wire_t], graph:nx.Graph):
                            phys_qumodes = [('qm', q[1]) for q in list_qubits]
                            #print(phys_qumodes)
                            avg_dist = 0
                            for q in phys_qumodes:
                                sum_dist = 0
                                for q2 in phys_qumodes:
                                    if q == q2:
                                        continue
                                    sum_dist += nx.shortest_path_length(graph, q, q2)
                                avg_dist += sum_dist / max(len(phys_qumodes), 1)
                            avg_dist /= len(phys_qumodes)
                            return avg_dist
                        while True:
                            log_qubits = [q for q in target.wires if q[0] == 'q']
                            phys_qubits = [current_mapping.get_physical_from_log(q) for q in log_qubits]
                            phys_qumodes = [('qm', q[1]) for q in phys_qubits]

                            loc_graph = current_mapping.coupling_graph

                            curent_avg_dist = get_avg_dist(phys_qubits, loc_graph)
                            #print("cur_avg_dist", curent_avg_dist)
                            if curent_avg_dist < 2.5 or is_connected(phys_qubits, loc_graph):
                                break

                            #print("hit")

                            center = min(current_mapping.coupling_graph.nodes, 
                                         key=lambda x : sum([math.pow(nx.shortest_path_length(loc_graph, x, y), 2) for y in phys_qumodes]))
                            center_comp = set()
                            if center in phys_qumodes:
                                for c in nx.connected_components(loc_graph.subgraph(phys_qumodes)):
                                    if center not in c:
                                        continue
                                    center_comp = c
                                    break

                            #next_swap = None
                            '''next_avg_dist = curent_avg_dist
                            #print(current_mapping)
                            #print(target)
                            #print(log_qubits)
                            #print(phys_qubits)
                            for pq in phys_qubits:
                                #print("pq", pq)
                                for n in current_mapping.coupling_graph.neighbors(('qm', pq[1])):
                                    #print("check", n,q)
                                    if n[0] == 'q':
                                        continue
                                    if n[1] >= pq[1]:
                                        continue
                                    if ('q',n[1]) in phys_qubits:
                                        continue
                                    #print("checked", n,q)
                                    if next_swap == None:
                                        next_swap = (('q', n[1]), pq)
                                        continue
                                    test_mapping = current_mapping.copy()
                                    test_mapping.swap_physical_qubits(('q', n[1]), pq)
                                    test_phys_qubits = [test_mapping.get_physical_from_log(q) for q in log_qubits]
                                    test_avg_dist = get_avg_dist(test_phys_qubits, test_mapping.coupling_graph)
                                    if test_avg_dist < next_avg_dist:
                                        next_swap = (('q', n[1]), pq)
                                        next_avg_dist = test_avg_dist'''
                            
                            moving_q = min([q for q in phys_qumodes if q not in center_comp], key=lambda x : nx.shortest_path_length(loc_graph, x, center))
                            target_q = min([x for x in loc_graph.neighbors(moving_q) if x not in phys_qumodes], key=lambda x: nx.shortest_path_length(loc_graph, x, center))

                            val = math.sqrt(math.pi/2)
                            qm1, qm2 = moving_q, target_q
                            qb1, qb2 = [('q', i) for _,i in (qm1, qm2)]
                            '''def make_swap(m1, m2):
                                gates:list[Gate] = []
                                gates.append(Gate('BS', [m1, m2], [utils.PiAngle(1), 0]))
                                gates.append(Gate('R', [m1], [utils.PiAngle(0.5)]))
                                gates.append(Gate('R', [m2], [utils.PiAngle(0.5)]))
                                return gates'''
                            def apply_cnot(q1, q2, m1, m2):
                                nodes:list[Node] = []
                                nodes.append(Gate('CD', [q1, m1], [val]))
                                nodes.append(Gate('h', [q2]))
                                nodes.append(Gate('BS', [m1, m2], [math.pi, math.pi/2]))
                                nodes.append(Gate('CD', [q2, m2], [val*1j]))
                                nodes.append(Gate('BS', [m1, m2], [math.pi, -math.pi/2]))
                                nodes.append(Gate('CD', [q1, m1], [-val]))
                                nodes.append(Gate('BS', [m1, m2], [math.pi, math.pi/2]))
                                nodes.append(Gate('CD', [q2, m2], [-val*1j]))
                                nodes.append(Gate('BS', [m1, m2], [math.pi, -math.pi/2]))
                                nodes.append(Gate('h', [q2]))
                                    
                                for node in nodes:
                                    out.append_node(node)
                                update_depth(nodes, multiple=True)
                                pass
                            apply_cnot(qb1, qb2, qm1, qm2)
                            apply_cnot(qb2, qb1, qm2, qm1)
                            apply_cnot(qb1, qb2, qm1, qm2)
                            current_mapping.swap_physical_qubits(qb1, qb2)
                            #print("qubit swaps:", qb1, qb2)


                    #print(out.wires)    

                    for n in target.map_to_coupling_graph(current_mapping, tsp_method=tsp_method).get_sequence():
                        #print(f"pauli| add node:{n}")
                        out.append_node(n)
                        update_depth(n)
                        tracked = tracked.union(n.ins.values())
                        
                else:
                    new_node = target.copy_disconnected()
                    new_node.wires = [current_mapping.get_physical_from_log(n) for n in new_node.wires]
                    out.append_node(new_node)
                    update_depth(new_node)
                    tracked = tracked.union(new_node.ins.values())

            tracked.add(target)

        if len(buffer) != 0:
            raise Exception(f"Improper Graph Structure: {this.wires}, {this.ins} -> {this.outs}\n buffer:{buffer}")
        
        return out
        

    def remove_commentbounds(this):
        for node in this.get_sequence():
            if node.nodeType == "commentLine":
                node.remove_from_circuit()

    def get_metrics(this):
        out = dict()
        seq = this.get_sequence()
        out['single op gate count'] = len([True for n in seq if len(n.wires) <= 1])
        out['multi op gate count'] = len(seq) - out['single op gate count']
        out['total gate count'] = len(seq)
        out['depth'] = max(this.depths().values())
        return out