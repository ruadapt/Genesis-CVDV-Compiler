# src.output_manager

from src.models import PauliString
import os

class OutputManager:
    """
    Manages the generation and output of CVDVQASM code.
    """
    def __init__(self, output_file=None):
        """
        Initialize the OutputManager.
        
        Args:
            output_file (str, optional): Path to the output file.
            instructions (list, optional): List of instructions to be added to the output file.
        """
        self.output_file = output_file
        self.instructions = []

    def set_output_file(self, filename):
        """Set the output file path."""
        self.output_file = filename

    def add_instruction(self, instruction):
        """Add a QASM instruction to the program."""
        self.instructions.append(instruction)

    def add_comment(self, comment):
        """Add a comment to the program."""
        self.instructions.append(f"// {comment}")
    
    def add_pauli_string(self, pauli_string_object: PauliString):
        """Add a PauliString to the program."""
        self.instructions.append(pauli_string_object.pauli_string_object.pauli_string)
    
    def write_to_file(self):
        """Write all accumulated instructions to the output file."""
        if not self.output_file:
            raise ValueError("Output file path not set")
        
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))

        with open(self.output_file, 'w') as f:
            # Write header
            # No header right now
            
            # Write instructions
            for instruction in self.instructions:
                f.write(instruction + ";\n")
