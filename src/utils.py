# src.utils

import os
import logging
import atexit
from typing import Union
from antlr4 import FileStream, CommonTokenStream
from generated.hamiltonianDSLLexer import hamiltonianDSLLexer
from generated.hamiltonianDSLParser import hamiltonianDSLParser
from src.error_listener import HamiltonianErrorListener
from src.hamiltonian_visitor import HamiltonianVisitor


def get_output_folder():
    """
    Return the output folder (located in the project root 'output' folder).
    Creates the folder if it doesn't exist.
    """
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def parse_file(file_path):
    """
    Parse the input file and return a tuple (ast_root, visitor).
    If syntax or semantic errors occur, print them and return (None, None).
    """
    try:
        # Create input stream with UTF-8 encoding.
        input_stream = FileStream(file_path, encoding="utf-8")
        
        # Initialize the lexer and add error listener.
        lexer = hamiltonianDSLLexer(input_stream)
        error_listener = HamiltonianErrorListener()
        lexer.removeErrorListeners()
        lexer.addErrorListener(error_listener)
        
        # Create token stream.
        token_stream = CommonTokenStream(lexer)
        
        # Initialize the parser and add error listener.
        parser = hamiltonianDSLParser(token_stream)
        parser.removeErrorListeners()
        parser.addErrorListener(error_listener)
        
        # Parse the input starting from the 'program' rule.
        tree = parser.program()
        
        # Check for syntax errors.
        if error_listener.hasErrors():
            print("Syntax Errors:")
            for error in error_listener.getErrors():
                print(error)
            return None, None
        
        # Visit the parse tree to build the AST.
        visitor = HamiltonianVisitor()
        ast_root = visitor.visit(tree)
        return ast_root, visitor

    except Exception as e:
        print(f"Error during parsing: {str(e)}")
        return None, None


def get_output_path(input_file_name, base_name, counter=None):
    """
    Generate output file path based on base name and counter.
    If counter is None, generate a single output file name with the pattern:
      {input_file_name}_{base_name}.cvdvqasm
    Otherwise, generate:
      {base_name}({counter}).cvdvqasm
    """
    output_dir = get_output_folder()
    if not counter:
        return os.path.join(output_dir, f"{input_file_name}_{base_name}.cvdvqasm")
    return os.path.join(output_dir, f"{base_name}({counter}).cvdvqasm")


def is_a_valid_file_path_format(output_path: Union[str, os.PathLike]) -> bool:
    """
    Check whether a given path is a syntactically valid file path.
    """
    path = str(output_path).strip()
    if not path:
        return False

    base = os.path.basename(path).strip()
    if not base:
        return False  # likely a directory path (ends with slash)

    return True


def insert_prefix(output_path: Union[str, os.PathLike], prefix: str) -> str:
    """
    Insert a prefix to the file name in the output path.
    
    If the output_path refers to a file, this function prepends `prefix` to the file name.
    If it does not appear to be a file (e.g., ends with a slash), the original path is returned.
    
    Args:
        output_path (str or PathLike): The original file path.
        prefix (str): The prefix string to insert before the file name.
    
    Returns:
        str: The updated path with the prefixed file name, or the original path if not a file.
    """
    output_path = str(output_path)
    if os.path.basename(output_path):  # Only apply if it looks like a file path
        dir_name = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        return os.path.join(dir_name, prefix + base_name)
    else:
        return output_path

def reconfigure_logger_for_input(input_file):
    """
    Reconfigure the FileHandler for the hamiltonian_visitor logger so that
    the log file name is derived from the input file and stored in the output folder.
    Ensures old handlers are closed to prevent resource warnings.
    """

    # Determine output directory and log file path
    output_dir = get_output_folder()
    input_base = os.path.splitext(os.path.basename(input_file))[0]
    log_file_path = os.path.join(output_dir, f"{input_base}_debug.log")

    # Fetch the specific logger
    logger = logging.getLogger("src.hamiltonian_visitor")

    # Remove and close existing FileHandlers to avoid unclosed file warnings
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    # Create and configure a new FileHandler
    new_file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    new_file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S'
    )
    new_file_handler.setFormatter(formatter)
    logger.addHandler(new_file_handler)

    # Also output to console for user visibility
    console_logger = logging.getLogger(__name__)
    console_logger.info(f"Logging debug information to: {log_file_path}")

    # Register shutdown to close all handlers on exit
    atexit.register(logging.shutdown)


def is_only_real_part(coef: complex) -> bool:
    return coef.imag == 0


def get_raw_coefficient(coef: complex, is_only_real_part: bool = False, multi_negative_i: bool = False) -> str:
    """
    Get the raw text of the coefficient without parentheses and spaces.
    """    
    if is_only_real_part:
        coef = coef.real
    if multi_negative_i:
        coef = -1j * coef
    coef_s = str(coef).replace(" ", "")
    if coef_s.startswith("(") and coef_s.endswith(")"):
        return coef_s[1:-1]
    return coef_s

def transform_fullpath_to_relative(fullpath: str, current_dir: str = os.getcwd()) -> str:
    """
    Transform a full path to a relative path.
    """
    return os.path.relpath(fullpath, current_dir)
