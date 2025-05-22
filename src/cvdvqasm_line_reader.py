# src.cvdvqasm_reader

from src.output_manager import OutputManager
import os
import warnings
import src.pattern_match_apply as pattern_match_apply


class CVDVQASMLineReader:
    def __init__(self, file_path: str, output_manager: OutputManager, is_print_debug_comments: bool = False, qbit_count:int = 0):
        self.file_path = file_path
        self.output_manager = output_manager
        self.is_print_debug_comments = is_print_debug_comments
        self.qubit_count = qbit_count
        self.max_pauli_length = 0

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Update qubit_count based on Pauli string length
        with open(self.file_path, 'r') as f:
            for line in f:
                if self._is_pauli_string(line):
                    # Extract the Pauli string from the line
                    if ":" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            pauli_str = parts[1].strip()
                            # Remove the trailing semicolon and any whitespace
                            pauli_str = pauli_str.strip().rstrip(';').strip()
                            self.max_pauli_length = max(self.max_pauli_length, len(pauli_str))
                            
        self.qubit_count += self.max_pauli_length    
                        
    def __iter__(self):
        with open(self.file_path, 'r') as f:
            for line_number, line in enumerate(f, start=1):
                yield line.strip(), line_number
    
    def run(self):
        for line, line_number in self:
            self._process_line(line, line_number)
    
    def _is_pauli_string(self, line: str) -> bool:
        return line.startswith("pauli") and line.endswith(";")
    
    def _is_bosonic_string(self, line: str) -> bool:
        return line.startswith("bosonic") and line.endswith(";")
    
    def _is_hybrid_string(self, line: str) -> bool:
        return line.startswith("hybrid") and line.endswith(";")
    
    def _is_comment_string(self, line: str) -> bool:
        return line.startswith("//")
    
    def _is_legal_line(self, line: str) -> bool:
        return self._is_pauli_string(line) or self._is_bosonic_string(line) or self._is_hybrid_string(line) or self._is_comment_string(line)
    
    def _process_pauli_string(self, line: str, line_number: int) -> None:
        if self.is_print_debug_comments:
            self.output_manager.add_comment(f"Pauli String Line {line_number}")
        
        new_line = line.replace(":", "")
        #new_line = new_line.replace("j", "i")
        new_line = new_line[:-1]
        self.output_manager.add_instruction(new_line)
        self.output_manager.write_to_file()
        
    def _process_exp_string(self, line: str, line_number: int) -> None:
        if self.is_print_debug_comments:
            self.output_manager.add_comment(f"Bosonic/Hybrid String Line {line_number}")

        _, input_line = line.split(maxsplit=1)
        input_line = input_line.strip(";")
        
        output = pattern_match_apply.main(input_line, self.qubit_count)
        if output == None:
            if self.is_print_debug_comments:
                self.output_manager.add_comment(f"Line {line_number}: pattern {input_line} not matched, ")
            print(f"Line {line_number}: pattern {input_line} not matched, ")
        self.qubit_count = output[0].index_counters['qubit']
        for line in output[1]:
            self.output_manager.add_instruction(line)
        self.output_manager.write_to_file()
    
    def _process_line(self, line: str, line_number: int) -> None:
        if not self._is_legal_line(line):
            warnings.warn(f"Illegal line {line_number}: {line}")
        elif self._is_pauli_string(line):
            self._process_pauli_string(line, line_number)
        elif self._is_bosonic_string(line) or self._is_hybrid_string(line):
            self._process_exp_string(line, line_number)
