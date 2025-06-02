# Run this file from the folder containing 'tests', as following:
# python3 -m tests.single_testcase.test_single_testcase > tests/single_testcase/test_single_testcase_outlog.txt

import src.pattern_match_apply as pattern_match_apply, src.parser as parser, src.graph_mapping as graph_mapping
from timeit import default_timer as timer
import math

for i in range(1,12):
    in_file_path = f"tests/single_testcase/testcase{i}.afe"
    out_file_path = f"tests/single_testcase/testcase{i}_decomp.txt"
    out_cvdv_path = f"tests/single_testcase/testcase{i}_gates.cvdv"
    
    try:
        fp = open(in_file_path, 'r')
    except:
        continue

    in_str = fp.read()
    fp.close()

    print(in_str)
    start_time = timer()
    out_env, out_list, rule_hits = pattern_match_apply.main(in_str, 0, write_ruleslist="tests/single_testcase/ruleslist.txt")
    print(rule_hits)

    fp2 = open(out_file_path, 'w')
    for l in out_list:
        fp2.write(l + "\n")
    fp2.close()



    in_circ = parser.read_qasm(out_file_path)

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

                parser.write_qasm(out_circ, out_cvdv_path)

                total_time = timer() - start_time
                print("time:", total_time)
                print(max(out_circ.depths().values()))



    pass