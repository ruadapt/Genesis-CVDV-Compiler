# Run this file from the folder containing 'tests', as following:
# python3 -m tests.pauli_map_test.test_pauli_map > tests/pauli_map_test/test_pauli_map_outlog.txt

import src.parser as parser, src.graph_mapping as graph_mapping
import math
from timeit import default_timer as timer

in_file_path = f"tests/pauli_map_test/pauli.cvdvqasm"
out_file_path = f"tests/pauli_map_test/pauli_out.cvdvqasm"



in_circ = parser.read_qasm(in_file_path)

seq = in_circ.get_sequence()
#print(in_circ.wires)

square_len = math.ceil(math.sqrt(max([l for _,l in in_circ.wires])+1))
square_graph = graph_mapping.make_rectangle_graph(square_len, square_len)
for p in ('depth_sum',):
    for m in ('simulated_annealing_tsp',):
        for q in (False,):
            q_str = 'qbmov' if q else 'qbstuck'
            print(f"mapping {in_file_path}_{p}_{m}_{q_str}")

            start = timer()
            out_circ = in_circ.map_to_graph(square_graph, pauli_mapping_order=p,tsp_method=m, qubit_swaps=q)

            parser.write_qasm(out_circ, out_file_path)



pass