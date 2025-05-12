# src.main
# python3 -m src.decomp_molecule --batch-json molecule_small_batch_jobs.json
import os
import argparse
import json
import time
from src.utils import *
from src.circ_utils import *
from src.output_manager import OutputManager
import src.parser as parser
import src.graph_mapping as graph_mapping
from timeit import default_timer as timer

__algorithm_settings = [
    ('depth_sum', 'christofides', False),
    ('depth_sum', 'threshold_accepting_tsp', False),
    ('depth_sum', 'threshold_accepting_tsp', True)
]

def run_single(input_file: str, debug_mode: bool, output_override: str = None, stats_mode: bool = False, stats_file: str = "molecule_compilation_stats.json") -> None:
    """
    Process a single input file: parse, transform, interpret, and write outputs.
    """
    print(f"Parsing input file: {input_file}")
    start_time = time.time()

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Step3: Apply the mapping program to the logical cvdvqasm file
    if stats_mode:
        try:
            f = open(stats_file)
            out = json.load(f)
            f.close()
        except FileNotFoundError:
            out = dict()


    in_circ = parser.read_pauli_format(input_file)

    seq = in_circ.get_sequence()
    if stats_mode:
        pauli_count = len([True for n in seq if n.nodeType == "pauliNode"])
        non_pauli_count = len(seq) - pauli_count
        q_wires = [q for t,q in in_circ.wires if t=='q']
        qm_wires = [q for t,q in in_circ.wires if t=='qm']
        intermediate_qbit_count = max(q_wires) + 1 if q_wires else 0
        intermediate_qmode_count = max(qm_wires) + 1 if qm_wires else 0

    square_len = math.ceil(math.sqrt(max([l for _,l in in_circ.wires])+1))
    square_graph = graph_mapping.make_rectangle_graph(square_len, square_len)

    for p,m,q in __algorithm_settings:
        q_str = 'qbmov' if q else 'qbstuck'
        print(f"Mapping {input_file}_{p}_{m}_{q_str}")

        if output_override:
            physical_output_path = output_override + f"_{p}_{m}_{q_str}"
        else:
            physical_output_path = get_output_path(base_name + f"_{p}_{m}_{q_str}", 'Result')

        if not os.path.exists(os.path.dirname(physical_output_path)):
            os.makedirs(os.path.dirname(physical_output_path))

        mapping_start = time.time()
        out_circ = in_circ.map_to_graph(square_graph, pauli_mapping_order=p,tsp_method=m, qubit_swaps=q)

        if stats_mode:
            out[f"{input_file}_{p}_{m}_{q_str}"] = out_circ.get_metrics()
            out[f"{input_file}_{p}_{m}_{q_str}"]['matching_time'] = mapping_start - start_time
            out[f"{input_file}_{p}_{m}_{q_str}"]['mapping_time'] = time.time() - mapping_start
            out[f"{input_file}_{p}_{m}_{q_str}"]['total_time'] = time.time() - start_time
            out[f"{input_file}_{p}_{m}_{q_str}"]['pstrings_count'] = pauli_count
            out[f"{input_file}_{p}_{m}_{q_str}"]['non_pstring_initial_gate_count'] = non_pauli_count
            out[f"{input_file}_{p}_{m}_{q_str}"]['intermediate_qbit_count'] = intermediate_qbit_count
            out[f"{input_file}_{p}_{m}_{q_str}"]['intermediate_qmode_count'] = intermediate_qmode_count
            
            if not os.path.exists(os.path.dirname(stats_file)):
                os.makedirs(os.path.dirname(stats_file))

            fp = open(stats_file, 'w')
            json.dump(out, fp, indent=4)
            fp.close()
        parser.write_qasm(out_circ, physical_output_path)


        #------
        print(f"Finish all parsing, decomposition, and mapping for input file: {input_file}")
    else:
        print(f"No result assignments found in {input_file}")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Parse input file(s) and generate QASM code with optional batch and debug modes."
    )
    # Positional input for single-file mode
    parser.add_argument("input_file", nargs="?", help="Path to a single input file")
    # JSON batch file for multiple jobs
    parser.add_argument("--batch-json", help="Path to a JSON file listing batch jobs")
    parser.add_argument("--debug", action="store_true", help="Enable debug comments")
    parser.add_argument("--stats", action="store_true", help="Enable stats collection")
    args = parser.parse_args()

    # Determine mode: batch vs single
    if args.batch_json and args.input_file:
        parser.error("Cannot use 'input_file' and '--batch-json' together.")
    if not args.batch_json and not args.input_file:
        parser.error("Either 'input_file' or '--batch-json' must be provided.")

    # Build job list
    jobs = []
    if args.batch_json:
        with open(args.batch_json, 'r') as jf:
            jobs = json.load(jf)  
    else:
        jobs = [{"input_file": args.input_file, "debug": args.debug}]
    
    stats_mode = args.stats
    stats_file = f"mapping_stats_{time.strftime('%Y%m%d_%H%M%S')}.json"
    # Process each job independently
    start_time = time.time()
    i = 1
    for job in jobs:
        print(f"\nProcessing job {i} of {len(jobs)}")
        in_file = job.get("input_file")
        dbg = args.debug
        out_override = job.get("output_file")
        run_single(in_file, dbg, out_override, stats_mode, stats_file)
        i += 1

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time} seconds")
    
if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# Sample JSON batch file: batch_jobs.json
# ----------------------------------------------------------------------------
# [
#    {
#        "input_file": "../Qubit_hamiltonians_2/LiH/BK_LiH_sto3g_4_electrons_12_spin_orbitals_Hamiltonian_631_paulis.txt",
#        "output_file": "output/small_molecule_benchmark/BK_LiH_sto3g_4_electrons_12_spin_orbitals_Hamiltonian_631_paulis.cvdvqasm"
#    },
#    {
#        "input_file": "../Qubit_hamiltonians_2/LiH/JW_LiH_sto3g_4_electrons_12_spin_orbitals_Hamiltonian_631_paulis.txt",
#        "output_file": "output/small_molecule_benchmark/BK_LiH_sto3g_4_electrons_12_spin_orbitals_Hamiltonian_631_paulis.cvdvqasm"
#    }
#]


#~~~~~~~~~~~~~~~~~~~~