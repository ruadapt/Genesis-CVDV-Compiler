# src.circ_utils

import math
import networkx as nx
from typing_extensions import Self, Any, Literal #For annotations only

wire_t = tuple[Literal['q']|Literal['qm'],int]

class PiAngle():
    def __init__(this, piMult:float):
        this.mult = piMult
    
    def get_normalized_mult(this):
        out = this.mult
        while out > 1:
            out -= 2
        while out <= -1:
            out += 2
        return out

    def __add__(this, other):
        if other == 0:
            return this
        if other.__class__ == PiAngle:
            other:PiAngle
            out = PiAngle(this.mult + other.mult)
            out.mult = out.get_normalized_mult()
            return out
        return (math.pi * this.mult) + other
    
    def __radd__(this, other):
        return this.__add__(other)

    def __neg__(this):
        return PiAngle(-this.mult)

    def __sub__(this, other):
        return this + (-other)

    def __rsub__(this, other):
        return other + (-this)

    def __mul__(this, other):
        return PiAngle(this.mult * other)

    def __rmul__(this, other):
        return PiAngle(other * this.mult)

    def __truediv__(this, other):
        return PiAngle(this.mult / other)

    def __rtruediv__(this, other):
        return other / (this.mult * math.pi)

    def __eq__(this, other):
        if other.__class__ == PiAngle:
            other:PiAngle = other
            return this.get_normalized_mult() == other.get_normalized_mult()
        return this.get_normalized_mult() * math.pi == other

    def __neq__(this, other):
        return not (this == other)
    
    def __repr__(this) -> str:
        return f"{this.mult}*pi"
    
    def __float__(this):
        return this.mult * math.pi
    
    def __str__(this):
        return this.__repr__()


def make_rectangle_graph(w, h):
    graph = nx.Graph()
    for i in range(h):
        for j in range(w-1):
            graph.add_edge(('qm',i*w+j), ('qm',i*w+j+1))
    for i in range(h-1):
        for j in range(w):
            graph.add_edge(('qm',i*w+j), ('qm',(1+i)*w+j))
    for i in range(w*h):
        graph.add_edge(('qm',i), ('q',i))

    return graph


class Mapping():
    def __init__(this, coupling_graph:nx.Graph):
        this.coupling_graph = coupling_graph
        this.log_qubits:list[wire_t] = []
        this.phys_qubits:list[wire_t] = []

    def map_qubit(this, logical_qubit:wire_t, physical_qubit:wire_t):
        if logical_qubit in this.log_qubits:
            raise Exception()
        if physical_qubit in this.phys_qubits:
            raise Exception()
        this.log_qubits.append(logical_qubit)
        this.phys_qubits.append(physical_qubit)

    def unmap_logical_qubit(this, logical_qubit:wire_t):
        if logical_qubit not in this.log_qubits:
            raise Exception("???")
        index = this.log_qubits.index(logical_qubit)
        this.log_qubits.pop(index)
        this.phys_qubits.pop(index)

    def get_logical_from_phys(this, physical_qubit:wire_t):
        try:
            index = this.phys_qubits.index(physical_qubit)
            return this.log_qubits[index]
        except ValueError:
            return None
        
    def get_physical_from_log(this, logical_qubit:wire_t):
        try:
            index = this.log_qubits.index(logical_qubit)
            return this.phys_qubits[index]
        except ValueError:
            return None

    def swap_physical_qubits(this, phys_qubit_1:wire_t, phys_qubit_2:wire_t):
        phys1_mapped = this.get_logical_from_phys(phys_qubit_1) != None
        phys2_mapped = this.get_logical_from_phys(phys_qubit_2) != None

        if phys1_mapped == True:
            index1 = this.phys_qubits.index(phys_qubit_1)
        if phys2_mapped == True:
            index2 = this.phys_qubits.index(phys_qubit_2)
        if phys1_mapped == True:
            this.phys_qubits[index1] = phys_qubit_2
        if phys2_mapped == True:
            this.phys_qubits[index2] = phys_qubit_1

    def map_physical_qubit_at_logical(this, logical_qubit:wire_t, physical_qubit:wire_t):
        if logical_qubit not in this.log_qubits:
            raise Exception()
        this.phys_qubits[this.log_qubits.index(logical_qubit)] = physical_qubit

    def copy(this):
        out = Mapping(this.coupling_graph)
        out.log_qubits = this.log_qubits.copy()
        out.phys_qubits = this.phys_qubits.copy()
        return out
    
    def __eq__(this, value: object) -> bool:
        if value.__class__ != Mapping:
            return False
        value:Mapping = value
        if this.coupling_graph != value.coupling_graph:
            return False
        this_pairs = list(sorted(zip(this.log_qubits, this.phys_qubits)))
        val_pairs = list(sorted(zip(value.log_qubits, value.phys_qubits)))

        return this_pairs == val_pairs 
    
    def __repr__(this) -> str:
        return f'mapping: {list(zip(this.log_qubits, this.phys_qubits))}'


if __name__ == '__main__':
    #testing, not important
    p = PiAngle(0.5)
    q = PiAngle(-0.2)
    r = PiAngle(1)
    print(p+q)
    print(p*0.8+5-q*0.56)
    print(p*2, p*3, p*4, p*5)
    print(p/6)
    print(-q + p*4 + q == 0)
    print(-r / 4, (-r+PiAngle(1)-PiAngle(1)) / 4)
    print((3*r) / 4, (3*r+PiAngle(1)-PiAngle(1)) / 4)
