# src.main
# python3 -m src.main --batch-json small_batch_jobs.json
import sys
import os
import logging
import argparse
import json
import time
import warnings
from src.utils import *
from src.circ_utils import *
from src.hamiltonian_visitor import pretty_print_ast_tree
from src.encoding import transform_QuantumOpNode_jw_encoding
from src.output_manager import OutputManager
from src.interpreter import Interpreter
from src.cvdvqasm_line_reader import CVDVQASMLineReader
import src.parser as parser
import src.circuits as circuits
import src.graph_mapping as graph_mapping
import json, os
from timeit import default_timer as timer

def run_single(input_file: str, debug_mode: bool, output_override: str = None, stats_mode: bool = False, stats_file: str = "compilation_stats.json", clean_mode: bool = False) -> None:
    """
    Process a single input file: parse, transform, interpret, and write outputs.
    """
    print(f"Parsing input file: {input_file}")
    start_time = time.time()
    # Reconfigure logger for this input
    reconfigure_logger_for_input(input_file)

    # Parse the input file and construct AST
    ast_root, visitor = parse_file(input_file)
    if ast_root is None or visitor is None:
        print(f"Failed to parse {input_file}")
        return

    # Apply JW encoding if applicable
    transform_QuantumOpNode_jw_encoding(ast_root)

    # Pretty-print AST structure to debug log
    pretty_print_ast_tree(ast_root)

    # Initialize interpreter without output manager (set per assignment)
    interpreter = Interpreter(visitor.symbol_table, None, is_print_debug_comments=debug_mode)

    # Process each top-level result assignment
    if visitor.result_assignments:
        print(f"Found {len(visitor.result_assignments)} result(s) in {input_file}")
        counter_map = {}
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        for identifier in visitor.result_assignments:
            count = counter_map.get(identifier, 0)
            counter_map[identifier] = count + 1

            # Step0.5: Prepare the intermediate cvdvqasm file
            # Determine output path: override via JSON or default naming
            if output_override is not None:
                intermediate_output_path = insert_prefix(output_override, "intermediate_")
            else:
                intermediate_output_path = insert_prefix(get_output_path(base_name, identifier, count), "intermediate_")
            print(f"Writing intermediate result '{identifier}' to: {transform_fullpath_to_relative(intermediate_output_path)}")
            intermediate_output_manager = OutputManager(intermediate_output_path)
            intermediate_output_manager.add_comment(f"Generated for: {identifier}")
            intermediate_output_manager.write_to_file() 

            # Step1: Apply Hamiltonian Grammar Parser to interpret the AST
            interpreter.output_manager = intermediate_output_manager
            interpreter.interpret(ast_root, identifier)
            
            # Step1.5: Prepare the logical cvdvqasm file
            if output_override is not None:
                logical_output_path = insert_prefix(output_override, "logical_")
            else:
                logical_output_path = insert_prefix(get_output_path(base_name, identifier, count), "logical_")
            print(f"Writing logical result '{identifier}' to: {transform_fullpath_to_relative(logical_output_path)}")
            logical_output_manager = OutputManager(logical_output_path)
            logical_output_manager.add_comment(f"Generated for: {identifier}")
            logical_output_manager.write_to_file()
            
            # Step2: Apply the line reader to the intermediate cvdvqasm file
            line_reader = CVDVQASMLineReader(intermediate_output_path, logical_output_manager, is_print_debug_comments=debug_mode)
            line_reader.run()
            
            # Step2.5: Prepare the physical cvdvqasm file
            if output_override is not None:
                physical_output_path = output_override
            else:
                physical_output_path = get_output_path(base_name, identifier, count)
            print(f"Writing final result '{identifier}' to: {transform_fullpath_to_relative(physical_output_path)}")
            physical_output_manager = OutputManager(physical_output_path)
            physical_output_manager.add_comment(f"Generated for: {identifier}")
            physical_output_manager.write_to_file()
            
            # Step3: Apply the mapping program to the logical cvdvqasm file
            if stats_mode:
                try:
                    f = open(stats_file)
                    out = json.load(f)
                    f.close()
                except FileNotFoundError:
                    out = dict()

            in_circ = parser.read_qasm(logical_output_path)

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
            for p in ('depth_sum',):
                for m in ('simulated_annealing_tsp',):
                    for q in (False,):
                        q_str = 'qbmov' if q else 'qbstuck'
                        print(f"Mapping {transform_fullpath_to_relative(logical_output_path)}_{p}_{m}_{q_str}")

                        mapping_start = time.time()
                        out_circ = in_circ.map_to_graph(square_graph, pauli_mapping_order=p,tsp_method=m, qubit_swaps=q)

                        if stats_mode:
                            out[f"{logical_output_path}_{p}_{m}_{q_str}"] = out_circ.get_metrics()
                            out[f"{logical_output_path}_{p}_{m}_{q_str}"]['matching_time'] = mapping_start - start_time
                            out[f"{logical_output_path}_{p}_{m}_{q_str}"]['mapping_time'] = time.time() - mapping_start
                            out[f"{logical_output_path}_{p}_{m}_{q_str}"]['total_time'] = time.time() - start_time
                            out[f"{logical_output_path}_{p}_{m}_{q_str}"]['pstrings_count'] = pauli_count
                            out[f"{logical_output_path}_{p}_{m}_{q_str}"]['non_pstring_initial_gate_count'] = non_pauli_count
                            out[f"{logical_output_path}_{p}_{m}_{q_str}"]['intermediate_qbit_count'] = intermediate_qbit_count
                            out[f"{logical_output_path}_{p}_{m}_{q_str}"]['intermediate_qmode_count'] = intermediate_qmode_count
                            fp = open(stats_file, 'w')
                            json.dump(out, fp, indent=4)
                            fp.close()
                        parser.write_qasm(out_circ, physical_output_path)

            # Last step: Delete all the intermediate files and notify the user
            if clean_mode:
                try:
                    os.remove(intermediate_output_path)
                    os.remove(logical_output_path)
                    print(f"Successfully deleted intermediate and logical cvdvqasm files for {input_file}")
                except Exception as e:
                    warnings.warn(f"Error deleting intermediate and logical cvdvqasm files for {input_file}: {e}")
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
    parser.add_argument("-o", "--output", help="Output file path")
    # JSON batch file for multiple jobs
    parser.add_argument("--batch-json", help="Path to a JSON file listing batch jobs")
    parser.add_argument("--debug", action="store_true", help="Enable debug comments")
    parser.add_argument("--stats", action="store_true", help="Enable stats collection")
    parser.add_argument("--clean", action="store_true", help="Only output the final result")
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
    stats_file = f"compilation_stats_{time.strftime('%Y%m%d_%H%M%S')}.json"
    clean_mode = args.clean
    # Process each job independently
    start_time = time.time()
    i = 1
    for job in jobs:
        print(f"\nProcessing job {i} of {len(jobs)}")
        input_file = job.get("input_file")
        dbg = args.debug
        out_override = job.get("output_file")
        if not is_a_valid_file_path_format(input_file):
            print(f"Invalid input file format: {input_file}")
            continue
        if out_override is not None and not is_a_valid_file_path_format(out_override):
            print(f"Invalid output file format: {out_override}")
            continue
        if out_override is None and args.output is not None:
            out_override = args.output
        
        run_single(input_file, dbg, out_override, stats_mode, stats_file, clean_mode)
        i += 1

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time} seconds")
    
if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# Sample JSON batch file: batch_jobs.json
# ----------------------------------------------------------------------------
# [
#   {
#     "input_file": "examples/ham1.hdsl",    # Path to the DSL source
#     "output_file": "out/ham1_result.qasm", # Desired output path
#   },
#   {
#     "input_file": "examples/ham2.hdsl",
#     "output_file": "out/ham2_result.qasm"
#   }
# ]
