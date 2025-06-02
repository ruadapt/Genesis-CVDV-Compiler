import src.circuits as circuits
import src.graph_mapping as graph_mapping
import src.circ_utils as utils
from typing import Any
from collections import deque
import math, networkx as nx
import re

def read_coupling_graph(filename:str):
    pass

def read_pauli_format(filename:str):
    out:circuits.Circuit = circuits.Circuit(wires=[])
    with open(filename, 'r') as fp:
        for line in fp:
            phrases = line.split()
            if len(phrases) != 2:
                continue
            paulistr, displacement = phrases

            #print(displacement)
            match = re.search(r"\((-?[\d.]*)(?:e(-?[\d.]*))?\+?(-?[\d.]*)j\)", displacement)
            #print(match.groups())
            dis_val = float(match[1])*math.pow(10, float(match[2]) if match[2] != None else 0)+float(match[3])*1j

            for i,_ in enumerate(paulistr):
                if ('q', i) not in out.wires:
                    out.extend_wires(('q', i))
                if ('qm', i) not in out.wires:
                    out.extend_wires(('qm', i))

            out.append_node(circuits.PauliNode(paulistr, dis_val))
    return out

def read_qasm(filename:str):
    out:circuits.Circuit = circuits.Circuit(wires=[])
    with open(filename, 'r') as fp:

        for line in fp:
            comment_index = line.find(r"//")
            if comment_index != -1:
                line = line[:comment_index]
            comment_index = line.find(r"#")
            if comment_index != -1:
                line = line[:comment_index]
            
            if line.strip() == "":
                continue

            command, qbits = line.strip().strip(";").split(" ", 1)
            if '(' in command and ')' not in command and len(qbits) > 0:
                right_paren = qbits.index(")")
                command = command + qbits[:right_paren+1]
                qbits = qbits[right_paren+1:]
            cbits = None
            if qbits.find("->") != -1:
                qbits, cbits = qbits.split("->")


            if command == "include":
                continue
            if command == "OPENQASM":
                continue

            params = []
            if "(" in command:
                left_paren = command.index("(")
                right_paren = command.index(")")

                params = command[left_paren+1:right_paren].split(",")
                params = [math_str_to_float(p) for p in params]
                command = command[:left_paren]

            qbit_strs = [s.strip() for s in qbits.strip().split(',')]

            if command == "pauli":
                pstring = qbit_strs[0]
                '''real, imag = params[0]
                disp = float(real) + float(imag) * 1j'''
                disp = math_str_to_float(params[0])
                out.append_node(circuits.PauliNode(pstring, disp), force=True)
                continue

            if command in ("reserve", "release"):
                for qs in qbit_strs:
                    if ('qm',qs) not in out.wires:
                        if command == "reserve":
                            out.extend_wires(('qm',new_qmode_val))
                        else:
                            raise Exception("Ancilla qumode does not exist")
                continue

            qbit_vals = []
            qmode_vals = []
            for qs in qbit_strs:
                if qs[0:2] == 'q[':
                    new_qbit_val = qs.strip().strip('q[]')
                    try:
                        new_qbit_num_val = int(new_qbit_val)
                        qbit_vals.append(new_qbit_num_val)
                        if ('q',new_qbit_num_val) not in out.wires:
                            out.extend_wires(('q',new_qbit_num_val))
                    except ValueError:
                        qbit_vals.append(new_qbit_val)
                        if ('q',new_qbit_val) not in out.wires:
                            out.extend_wires(('q',new_qbit_val))
                elif qs[0:3] == 'qm[':
                    new_qmode_val = qs.strip().strip('qm[]')
                    try:
                        new_qmode_num_val = int(new_qmode_val)
                        qmode_vals.append(new_qmode_num_val)
                        if ('qm',new_qmode_num_val) not in out.wires:
                            out.extend_wires(('qm',new_qmode_num_val))
                    except ValueError:
                        qmode_vals.append(new_qmode_val)
                        if ('qm',new_qmode_val) not in out.wires:
                            out.extend_wires(('qm',new_qmode_val))

            if cbits != None:
                cbit_strs = cbits.strip().split(',')
                cbit_vals = []
                for cs in cbit_strs:
                    cbit_vals.append(int(cs.strip().strip('c[]')))

            if command == "qreg":
                out = circuits.Circuit(qubit_size=qbit_vals[0] if len(qbit_vals)>0 else 0,
                                        qumode_size=qmode_vals[0] if len(qmode_vals)>0 else 0)
                continue
            #print((command, [('q',i) for i in qbit_vals] + [('qm',i) for i in qmode_vals], params))
            out.append_node(circuits.Gate(command, [('q',i) for i in qbit_vals] + [('qm',i) for i in qmode_vals], params))
    return out

def write_qasm(circ:circuits.Circuit, filename:str):
    with open(filename, 'w') as fp:
        fp.write('CVDVQASM 1.0;\n')
        qbit_size = max([t for q,t in circ.wires if q=='q'])+1
        qmode_size = max([t for q,t in circ.wires if q=='qm'])+1
        fp.write(f"qreg q[{qbit_size}] qm[{qmode_size}];\n")
        for node in circ.get_sequence():
            fp.write(node.to_instr()+"\n")

def read_coupling_graph(filename:str):
    out:nx.Graph = nx.Graph()
    with open(filename, 'r') as fp:
        for line in fp:
            vertices = [s.strip() for s in line.split("-")]
            processed_vertices = []
            for v in vertices:
                if v[0:2] == 'q[':
                    new_qbit_val = v.strip().strip('q[]')
                    try:
                        new_qbit_num_val = int(new_qbit_val)
                        processed_vertices.append(('q',new_qbit_num_val))
                    except ValueError:
                        processed_vertices.append(('q',new_qbit_val))
                elif v[0:3] == 'qm[':
                    new_qmode_val = v.strip().strip('qm[]')
                    try:
                        new_qmode_num_val = int(new_qmode_val)
                        processed_vertices.append(('qm',new_qmode_num_val))
                    except ValueError:
                        processed_vertices.append(('qm',new_qmode_val))
            out.add_edge(processed_vertices[0], processed_vertices[1])
    
    return out

'''def read_hamiltonian(filename:str):
    out = hamiltonian.HamilNode('sum',None,[])
    with open(filename, 'r') as fp:

        for line in fp:
            comment_index = line.find(r"//")
            if comment_index != -1:
                line = line[:comment_index]
            comment_index = line.find(r"#")
            if comment_index != -1:
                line = line[:comment_index]
            
            if line.strip() == "":
                continue

            command, qbits = line.strip().strip(";").split(" ", 1)

    return out'''

def _read_hamiltonian_text(hamil_text:str):
    pass

def math_str_to_float(math_str):
    try:
        out = complex(math_str)
        return out
    except ValueError:
        pass
    #Below is WIP
    tokens:deque[tuple[str,Any]] = deque()
    ptr = 0
    while ptr < len(math_str):
        c:str = math_str[ptr]
        if c in "-*/^":
            tokens.append((c, None))
        elif c.isspace():
            continue
        elif c in "(":
            old_ptr = ptr
            while True:
                ptr += 1
                nesting = 1
                if ptr >= len(math_str):
                    raise Exception(f"Parsing Error : {math_str}")
                c2:str = math_str[ptr]
                if c2 == '(':
                    nesting += 1
                elif c2 == ')':
                    nesting -= 1
                if nesting >= 0:
                    break
            tokens.append(('num', math_str_to_float(math_str[old_ptr+1:ptr])))
        else:
            if c == "p":
                if len(math_str) - ptr >= 2 and math_str[ptr:ptr+2] == "pi":
                    tokens.append(("num", utils.PiAngle(1)))
                    ptr += 1
                else:
                    raise Exception(f"Parsing Error : {math_str}")
            elif c in "1234567890.":
                start_ptr = ptr
                while ptr < len(math_str)-1 and math_str[ptr+1] in "1234567890.":
                    ptr += 1
                tokens.append(("num", float(math_str[start_ptr:ptr+1])))
            elif c in "ij":
                if len(tokens) >= 1 and tokens[-1][0] == "num":
                    _, num = tokens.pop()
                    tokens.append(("num", num*1j))
                else:
                    tokens.append(("num", 1j))
            else:
                raise Exception(f"Parsing Error: {c}")
            if len(tokens) >= 2 and tokens[-2][0] == "-" and (len(tokens) == 2 or tokens[-3][0] not in ('num')) :
                _, num = tokens.pop()
                tokens.pop()
                tokens.append(("num", -num))
            
        ptr += 1
    #print(tokens)

    while len(tokens) > 1:
        if len(tokens) == 2 or len(tokens) <= 0:
            raise Exception(f"Parsing Error: {tokens}")
        if tokens[0][0] == tokens[2][0] == "num":
            _, num1 = tokens.popleft()
            op, _ = tokens.popleft()
            _, num2 = tokens.popleft()
            if op == "*":
                tokens.appendleft(("num",num1*num2))
            elif op == "/":
                tokens.appendleft(("num",num1/num2))
            else:
                raise Exception(f"Parsing Error: {tokens}")
            continue
        raise Exception(f"Parsing Error: {tokens}")
    
    return tokens[0][1]
