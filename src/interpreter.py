# src.interpreter

from src.models import *
from src.ast_nodes import *  
from src.output_manager import OutputManager
from typing import Any
import sys
import logging
import warnings
from src.utils import get_raw_coefficient, is_only_real_part
from src.serializer import serialize_pauli_ops_terms, serialize_multi_operator_terms_to_intermediate_hamiltonian

# Configure warnings to show all warnings
warnings.filterwarnings('always')

# Initialize logger for the interpreter module
logger = logging.getLogger(__name__)
logger = logging.getLogger("src.hamiltonian_visitor")
logger.setLevel(logging.DEBUG)
logger.handlers = []
logger.propagate = False 

class Interpreter:
    def __init__(self, symbol_table: dict[str, Any], output_manager: OutputManager, is_print_debug_comments: bool = False):
        self.symbol_table = symbol_table
        self.output_manager = output_manager
        self.logger = logger
        self.is_print_debug_comments = is_print_debug_comments
        self.is_sigma = False

    def interpret(self, ast_root: ASTNode, target_identifier=None) -> None:
        if isinstance(ast_root, AssignmentNode):
            self._traverse_ast(ast_root.expression)
        else:
            for child in self._get_children(ast_root):
                self.interpret(child)

    def _traverse_ast(self, node: ASTNode) -> None:
        def _unwrap_parens(n: ASTNode) -> ASTNode:
            while isinstance(n, ParenExpressionNode):
                n = n.expression
            return n

        def _handle_sum_over(acc_node: AccumulationNode, coeff: MultiOperatorTerms):
            if self.is_print_debug_comments:
                self.output_manager.add_comment(f"Sum_over node")
            terms_list = self._process_sum_over(acc_node, coeff)
            self.output_manager.write_to_file()
            for terms in terms_list:
                if terms.op_type is None:
                    warnings.warn(f"Warning: MultiOperatorTerms has no op_type: {terms}")

        def _extract_sum_and_coeff(node: ASTNode) -> Tuple[Optional[AccumulationNode], MultiOperatorTerms]:
            """Extract the Sum_over node and simplify any numeric coefficients preceding it."""
            node = _unwrap_parens(node)

            if isinstance(node, AccumulationNode) and node.op == AccumulationType.SUM_OVER:
                return node, MultiOperatorTerms([SingleOperatorTerm("coefficient", 1)])

            elif isinstance(node, UnaryMinusNode):
                sum_node, coeff = _extract_sum_and_coeff(node.expression)
                coeff.terms[0].coefficient *= -1
                return sum_node, coeff

            elif isinstance(node, BinaryOpNode):
                if node.op == BinaryOperator.DIV:
                    left_count = self._count_sum_over(node.left)
                    right_count = self._count_sum_over(node.right)
                    if right_count:
                        raise ValueError("Sum_over cannot appear in denominator.")

                left_count = self._count_sum_over(node.left)
                right_count = self._count_sum_over(node.right)

                if left_count and right_count:
                    raise ValueError("Cannot combine two Sum_over expressions with multiplication or division.")

                if left_count or right_count:
                    sum_node = node.left if left_count else node.right
                    coeff_node = node.right if left_count else node.left

                    sum_node = _unwrap_parens(sum_node)
                    coeff_node = _unwrap_parens(coeff_node)

                    sum_over, base_coeff = _extract_sum_and_coeff(sum_node)

                    # Attempt to evaluate coefficient node
                    if not self._is_pure_coefficient(coeff_node, {}):
                        raise ValueError("Coefficient must be a pure numeric expression.")
                    coeff_terms = self._get_ops_terms(coeff_node, {})
                    if coeff_terms.op_type != "coefficient":
                        raise ValueError("Coefficient expression did not evaluate to a numeric value.")
                    factor = coeff_terms.terms[0].coefficient

                    if node.op == BinaryOperator.DIV and left_count:
                        if factor == 0:
                            raise ZeroDivisionError("Division by zero in coefficient.")
                        factor = 1 / factor
                    
                    if not base_coeff.terms:
                        base_coeff.terms.append(SingleOperatorTerm(op_type="coefficient", coefficient=factor))
                    else:
                        base_coeff.terms[0].coefficient *= factor
                    return sum_over, base_coeff

                # If no Sum_over, and entire node is just a numeric coefficient
                if self._is_pure_coefficient(node, {}):
                    return None, self._get_ops_terms(node, {})

                raise ValueError("Expected a Sum_over expression or a numeric constant, but found a non-coefficient expression without Sum_over.")

            raise ValueError("Unsupported AST structure encountered.")

        node = _unwrap_parens(node)

        if isinstance(node, BinaryOpNode) and node.op in {BinaryOperator.ADD, BinaryOperator.SUB}:
            self._traverse_ast(node.left)

            if isinstance(node.right, AccumulationNode) or self._count_sum_over(node.right) > 0:
                # Extract Sum_over and coefficient
                sum_node, coeff_terms = _extract_sum_and_coeff(node.right)
                if node.op == BinaryOperator.SUB:
                    # SUB: Negate the coefficient sign
                    for term in coeff_terms.terms:
                        term.coefficient *= -1
                _handle_sum_over(sum_node, coeff_terms)
            else:
                self._traverse_ast(node.right)

            return

        sum_over_count = self._count_sum_over(node)
        if sum_over_count == 0:
            if self._is_pure_coefficient(node, {}):
                # Pure coefficient, no further processing needed
                return
            else:
                raise ValueError("Expressions without Sum_over must be pure coefficients.")

        if sum_over_count > 1:
            raise ValueError("Multiple Sum_over expressions detected in a context that requires only one.")

        # Extract Sum_over and calculate complete coefficient
        sum_node, coeff_terms = _extract_sum_and_coeff(node)
        if sum_node is None:
            # Only coefficient, no Sum_over, no processing needed
            return
        _handle_sum_over(sum_node, coeff_terms)


    # Helper function to get all children of the given AST node.
    def _get_children(self, node: ASTNode) -> list:
        """
        Return all child nodes of the given AST node.
        For known node types, directly return their children to avoid traversing irrelevant attributes.
        """
        if isinstance(node, BinaryOpNode):
            return [node.left, node.right]
        if isinstance(node, TensorProdNode):
            return node.factor_exprs
        if isinstance(node, AccumulationNode):
            # For accumulation nodes, return only the body_expr (range variables are handled separately)
            return [node.body_expr]
        # Default: traverse the node's __dict__
        children = []
        for attr in getattr(node, "__dict__").values():
            if isinstance(attr, ASTNode):
                children.append(attr)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, ASTNode):
                        children.append(item)
        return children


    # Helper function to check if the node is a pure coefficient expression
    def _is_pure_coefficient(self, node: ASTNode, current_assignment: dict) -> bool:
        """Check if the node is a pure coefficient expression"""
        if isinstance(node, (NumberLiteralNode, ImagNode)):
            return True
        if isinstance(node, IdentifierNode):
            raw = self.symbol_table.get(node.name)
            num = self._evaluate_parameter(raw if isinstance(raw, ASTNode) else NumberLiteralNode(raw), current_assignment)
            if isinstance(num, complex):
                if not is_only_real_part(num):
                    raise ValueError(f"Currently we don't support complex number as numeric coefficient for Sum_over expression.")
                else:
                    num = num.real
            return isinstance(num, (int, float))
        if isinstance(node, BinaryOpNode):
            return self._is_pure_coefficient(node.left, current_assignment) and self._is_pure_coefficient(node.right, current_assignment)
        if isinstance(node, UnaryMinusNode):
            return self._is_pure_coefficient(node.expression, current_assignment)
        if isinstance(node, ParenExpressionNode):
            return self._is_pure_coefficient(node.expression, current_assignment)
        return False


    # Helper function to check the number of Sum_over in the AST node(to check if the Sum_over is nested)
    def _count_sum_over(self, node: ASTNode) -> int:
        """Count the number of Sum_over in the AST node"""
        if isinstance(node, AccumulationNode) and node.op == AccumulationType.SUM_OVER:
            return 1
        return sum(self._count_sum_over(child) for child in self._get_children(node))
    
    
    # Helper function to evaluate an expression node as a parameter using the current assignment.
    def _evaluate_parameter(self, expr: ExpressionNode, current_assignment: dict) -> int:
        """
        Evaluate an expression node as a parameter using the current assignment.
        Supports number literals, identifiers, and binary operations.
        """
        if isinstance(expr, NumberLiteralNode):
            return expr.value
        elif isinstance(expr, IdentifierNode):
            # First check current loop variables
            if expr.name in current_assignment:
                return current_assignment[expr.name]

            value = self.symbol_table.get(expr.name)
            if value is None:
                raise ValueError(f"Identifier '{expr.name}' not found in symbol table")

            # Case 1: Direct numeric value
            if isinstance(value, (int, float)):
                return value

            # Case 2: AST node - evaluate recursively
            if isinstance(value, ASTNode):
                return self._evaluate_parameter(value, current_assignment)

            raise ValueError(f"Unable to evaluate parameter: {expr.name}")

        elif isinstance(expr, BinaryOpNode):
            left_val = self._evaluate_parameter(expr.left, current_assignment)
            right_val = self._evaluate_parameter(expr.right, current_assignment)
            op = expr.op
            if op == BinaryOperator.ADD:
                return left_val + right_val
            elif op == BinaryOperator.SUB:
                return left_val - right_val
            elif op == BinaryOperator.MUL:
                return left_val * right_val
            elif op == BinaryOperator.DIV:
                if right_val == 0:
                    raise ZeroDivisionError("Division by zero")
                return left_val / right_val
            elif op == BinaryOperator.POW:
                return left_val ** right_val
            else:
                raise ValueError(f"Unsupported binary operator: {op}")
        raise ValueError(f"Unsupported expression type in parameter: {type(expr)}")
    
    
    # Get all QuantumOpNode/Coefficient objects and return a MultiOperatorTerms object.
    def _get_ops_terms(self, node: ASTNode, current_assignment: dict) -> MultiOperatorTerms:
        """
        Recursively scan the AST to extract all QuantumOpNode objects and return a MultiOperatorTerms object.
        This object will be used in _process_sum_over for further processing.
        """
        current_terms = MultiOperatorTerms()
        
        # Case 1: Multiplication and Division
        if isinstance(node, BinaryOpNode) and node.op in {BinaryOperator.MUL, BinaryOperator.DIV}:
            left_terms  = self._get_ops_terms(node.left,  current_assignment) or None
            right_terms = self._get_ops_terms(node.right, current_assignment) or None

            if left_terms is None and right_terms is None:
                warnings.warn(f"Exception: Both left and right terms are None: {node}")
                return None

            if left_terms is None:
                warnings.warn(f"Exception: Left terms is None: {node}")
                return right_terms
            
            if right_terms is None:
                warnings.warn(f"Exception: Right terms is None: {node}")
                return left_terms

            if node.op == BinaryOperator.DIV:
                if left_terms.op_type == "coefficient" and right_terms.op_type == "coefficient":
                    num = left_terms.terms[0].coefficient / right_terms.terms[0].coefficient
                    return MultiOperatorTerms([ SingleOperatorTerm("coefficient", coefficient=num) ])
                elif left_terms.op_type != "coefficient" and right_terms.op_type == "coefficient":
                    terms = deepcopy(right_terms.terms)
                    for term in terms:
                        term.coefficient = 1 / term.coefficient
                    return left_terms * MultiOperatorTerms(terms=terms)
                else:
                    raise ValueError("Division over a QuantumOpNode is not supported yet")
            
            # If both sides are QuantumOpNode, multiply them together
            return left_terms * right_terms
                
        # Case 2: Addition and Subtraction
        if isinstance(node, BinaryOpNode) and node.op in {BinaryOperator.ADD, BinaryOperator.SUB}:
            left_terms = self._get_ops_terms(node.left, current_assignment)
            right_terms = self._get_ops_terms(node.right, current_assignment)
            if left_terms is None and right_terms is None:
                warnings.warn(f"Exception: Both left and right terms are None: {node}")
                return None
            if left_terms is None:
                warnings.warn(f"Exception: Left terms is None: {node}")
                return right_terms
            if right_terms is None:
                warnings.warn(f"Exception: Right terms is None: {node}")
                return left_terms
            
            # Leave the ADD and SUB calculation to models.MultiOperatorTerms
            if node.op == BinaryOperator.SUB:
                for right_term in right_terms.terms:
                    right_term.coefficient *= -1
            return left_terms + right_terms
        
        # Case 3: Number or Imaginary or Identifier or UnaryMinus
        if isinstance(node, NumberLiteralNode):
            return MultiOperatorTerms(terms=[SingleOperatorTerm(op_type="coefficient", coefficient=node.value)])
        
        if isinstance(node, ImagNode):
            return MultiOperatorTerms(terms=[SingleOperatorTerm(op_type="coefficient", coefficient=1j)])
        
        if isinstance(node, IdentifierNode) and not (hasattr(node, "op_type") and node.op_type.startswith("Pauli_")):            
            raw = self.symbol_table.get(node.name)
            num = self._evaluate_parameter(raw if isinstance(raw, ASTNode) else NumberLiteralNode(raw), current_assignment)
            
            if not isinstance(num, (int, float, complex)):
                raise ValueError(f"Unsupported identifier type: {type(num)}")
            
            return MultiOperatorTerms([SingleOperatorTerm("coefficient", coefficient=num)])
        
        if isinstance(node, UnaryMinusNode):
            # Get the MultiOperatorTerms for the inner expression
            inner_terms = self._get_ops_terms(node.expression, current_assignment)
            if inner_terms is None:
                return None
            # Create a MultiOperatorTerms with coefficient -1
            neg_one = MultiOperatorTerms([
                SingleOperatorTerm(op_type="coefficient", coefficient=-1)
            ])
            # Multiply -1 into inner_terms using __mul__
            return inner_terms * neg_one
        
        # Case 4: Parenthesized expression
        if isinstance(node, ParenExpressionNode):
            return self._get_ops_terms(node.expression, current_assignment)    
        
        # Case 5: Tensor Product over range
        if isinstance(node, AccumulationNode) and node.op == AccumulationType.TENSORPROD_OVER:
            range_vars = node.range_vars
            if len(range_vars) != 1:
                raise ValueError("TensorProd_over expects exactly one range variable.")
            rv = range_vars[0]
            if not rv.is_inline:
                raise ValueError("TensorProd_over expects an inline-declared range variable.")
            start = self._evaluate_parameter(rv.start_expr, current_assignment)
            end = self._evaluate_parameter(rv.end_expr, current_assignment)
            step = self._evaluate_parameter(rv.step_expr, current_assignment)
            if step == 0 or (start > end and step > 0) or (start < end and step < 0):
                raise ValueError("Invalid step value in inline TensorProd_over")
            for k_value in range(start, end, step):
                new_assignment = current_assignment.copy()
                new_assignment[rv.var_name] = k_value
                iteration_terms = self._get_ops_terms(node.body_expr, new_assignment)
                if current_terms.terms == []:
                    current_terms = iteration_terms
                else:
                    current_terms *= iteration_terms
            return current_terms

        # Case 6: Tensor Product Node
        if isinstance(node, TensorProdNode):
            for factor_expr in node.factor_exprs:
                factor_terms = self._get_ops_terms(factor_expr, current_assignment)
                if factor_terms is None or factor_terms.terms == []:
                    continue
                if current_terms.terms == []:
                    current_terms = factor_terms
                else:
                    current_terms *= factor_terms
            return current_terms
            
        # Case 7: Quantum Operator Node (Bosonic)
        if hasattr(node, "op_type") and node.op_type in {"BC", "BA"}:
            index = self._evaluate_parameter(node.indices[0], current_assignment)
            single_bosonic_term = BosonicOperator(node.op_type == "BC", index)
            single_ops_term = SingleOperatorTerm(op_type="bosonic", bosonic_term=[single_bosonic_term])
            current_terms+=MultiOperatorTerms(terms=[single_ops_term])
            return current_terms
        
        # Case 8: Quantum Operator Node (Fermionic)
        if hasattr(node, "op_type") and node.op_type in {"FA", "FC", "FN"}:
            index = self._evaluate_parameter(node.indices[0], current_assignment)
            if len(node.indices) == 2:
                sigma = self._evaluate_parameter(node.indices[1], current_assignment)
            else:
                sigma = None
            if node.op_type == "FN":
                single_fermionic_term = FermionicNumber(index, sigma)
            else:
                single_fermionic_term = FermionicOperator(node.op_type == "FC", index, sigma)
            single_ops_term = SingleOperatorTerm(op_type="fermionic", fermionic_term=[single_fermionic_term])
            current_terms+=MultiOperatorTerms(terms=[single_ops_term])
            return current_terms            
        
        # Case 9: Pauli Operator
        if hasattr(node, "op_type") and node.op_type.startswith("Pauli_"):
            pauli_type = node.op_type.split("_")[1]
            index = self._evaluate_parameter(node.indices[0], current_assignment)
            try:
                sigma = self._evaluate_parameter(node.indices[1], current_assignment)
            except:
                sigma = 0
            single_pauli_term = PauliOperator(pauli_type, index, sigma)
            single_ops_term = SingleOperatorTerm(op_type="pauli", pauli_term=[single_pauli_term])
            current_terms+=MultiOperatorTerms(terms=[single_ops_term])
            return current_terms

        # Exception: Unknown Quantum Operator Node
        if hasattr(node, "op_type"):
            if self.is_print_debug_comments: self.output_manager.add_comment(f"Unknown QuantumOpNode: {node.op_type}")
            raise ValueError(f"Unknown QuantumOpNode: {node.op_type}")
        
        return current_terms # Default: return empty MultiOperatorTerms if no QuantumOpNode is found
    
    
    # Process a sum_over accumulation expression
    def _process_sum_over(self, acc_node: AccumulationNode, coeff_factor: MultiOperatorTerms | None = None) -> List[MultiOperatorTerms]:
        """
        Process a sum_over accumulation expression:
        1. Iterate over all combinations of the range variables
        2. For each combination, evaluate body_expr to obtain multiple terms
        3. Each term is a list of PauliOperator objects corresponding to an individual PauliString
        4. Collect all terms and determine the maximum index found
        5. For each term, reconstruct a fixed-length string representation and output via output_manager
        """
        if coeff_factor is None:
            coeff_factor = MultiOperatorTerms(terms=[SingleOperatorTerm(op_type="coefficient", coefficient=1)])
        
        body_expr = acc_node.body_expr
        range_vars  = acc_node.range_vars

        # -------------------------------------------------
        # Phase 1: Record range expressions without evaluation
        # -------------------------------------------------
        temp_ranges: dict[str, tuple[ExpressionNode, ExpressionNode, ExpressionNode]] = {}
        for rv in range_vars:
            if rv.is_inline:
                temp_ranges[rv.var_name] = (rv.start_expr, rv.end_expr, rv.step_expr)
            else:
                value = self.symbol_table.get(rv.var_name)
                if not (isinstance(value, tuple) and len(value) == 3):
                    raise ValueError(f"Range var '{rv.var_name}' not declared or format error: {value}")
                temp_ranges[rv.var_name] = value   # Elements in value may be numbers or expressions

            if self.is_print_debug_comments:
                self.output_manager.add_comment(f"Range '{rv.var_name}' recorded (lazy-eval)")

        var_names = list(temp_ranges.keys())
        collected_multi_ops_terms = []

        # -------------------------------------------------
        # Phase 2: Depth-first expansion with lazy evaluation
        # -------------------------------------------------
        def _nested_loop(idx: int, assign: dict) -> None:
            if idx == len(var_names):                       
                # ---------- Leaf node ----------
                # Symmetry pruning (example: skip when i==j)
                non_sigma_vars = [v for v in var_names if v != "sigma"]
                if len(non_sigma_vars) == 2 and assign[non_sigma_vars[0]] == assign[non_sigma_vars[1]]:
                    if self.is_print_debug_comments:
                        self.output_manager.add_comment(f"Skip (sym) {non_sigma_vars[0]}={non_sigma_vars[1]}={assign[non_sigma_vars[0]]}")
                    return
                    
                if self.is_print_debug_comments:
                    self.output_manager.add_comment(f"Leaf assignment: {assign}")

                multi_terms = self._get_ops_terms(body_expr, assign)
                copy_multi_terms = deepcopy(multi_terms)
                try:
                    multi_terms = copy_multi_terms._simplify()
                except:
                    raise ValueError(f"Error simplifying multi_ops_terms: {multi_terms}")
                # Multiply the coefficient factor
                multi_terms *= coeff_factor

                collected_multi_ops_terms.append(multi_terms)
                if self.is_print_debug_comments:
                    self.output_manager.add_comment(f"Collected multi_ops_terms: {multi_terms}")
                return

            # ---------- Recursive expansion ----------
            vname = var_names[idx]
            start_e, end_e, step_e = temp_ranges[vname]

            # ⚠ Evaluate based on current assignment
            start = self._evaluate_parameter(start_e, assign)
            end   = self._evaluate_parameter(end_e,   assign)
            step  = self._evaluate_parameter(step_e,  assign)
            if step == 0:
                raise ValueError(f"Range '{vname}' step size is 0")

            for val in range(start, end, step):             # Python range is half-open; ensure DSL semantics match
                assign[vname] = val
                _nested_loop(idx + 1, assign)
            assign.pop(vname, None)                         # Clean up local variables

        _nested_loop(0, {})
        
        # Calculate the final PauliString length based on the maximum index and sigma presence
        # For sigma terms: length = (max_index + 1) * 2
        # For non-sigma terms: length = max_index + 1
        global_max_index = 0
        self.is_sigma = False
        for multi_terms in collected_multi_ops_terms:
            # if ANY Pauli term has sigma, then we put is_sigma to True
            if multi_terms._is_pauli_term_has_sigma():
                self.is_sigma = True
            # get the largest Pauli index from this MultiOperatorTerms
            global_max_index = max(global_max_index, multi_terms._largest_pauli_index())
        
        final_pauli_length = (global_max_index + 1) * 2 if self.is_sigma else global_max_index + 1
        
        # Initialize list to store non-Pauli operator terms
        cvdv_ops_terms = []
        
        # Process each multi-operator term based on its type:
        # - Pauli terms: Write as Pauli String directly to output file
        # - Bosonic/Hybrid terms: Store for intermediate Hamiltonian processing
        for multi_terms in collected_multi_ops_terms:
            if multi_terms.op_type == "pauli":
                if self.is_print_debug_comments: self.output_manager.add_comment("Pauli String Representations")
                serialized_pauli_ops_terms_str = serialize_pauli_ops_terms(multi_terms, final_pauli_length, self.is_sigma)
                for instr in serialized_pauli_ops_terms_str:
                    self.output_manager.add_instruction(instr)
            else:
                # Store non-Pauli terms for intermediate Hamiltonian processing
                cvdv_ops_terms.append(multi_terms)
                
        global_max_index = 0
        self.is_sigma = False
        temp_cvdv_ops_terms = cvdv_ops_terms.copy()
        cvdv_ops_terms = []
        removed_pauli_terms = []
        # Remove pauli single terms and store the removed pauli terms
        for multi_terms in temp_cvdv_ops_terms:
            global_max_index = max(global_max_index, multi_terms._largest_pauli_index())
            if multi_terms._is_pauli_term_has_sigma():
                self.is_sigma = True
            multi_terms, removed_pauli_term = multi_terms._remove_pauli_single_terms()
            removed_pauli_terms.extend(removed_pauli_term)
            cvdv_ops_terms.append(multi_terms)
            
        # Recalculate the final PauliString length based on the maximum index and sigma presence
        final_pauli_length = global_max_index * 2 if self.is_sigma else global_max_index + 1
        # Iterate over the removed pauli terms and write them as Pauli String Representations
        for removed_pauli_term in removed_pauli_terms:
            if self.is_print_debug_comments: self.output_manager.add_comment("Pauli String Representations")
            serialized_pauli_ops_terms_str = serialize_pauli_ops_terms(removed_pauli_term, final_pauli_length, self.is_sigma)
            for instr in serialized_pauli_ops_terms_str:
                self.output_manager.add_instruction(instr)
                   
        # Process stored non-Pauli terms to generate intermediate Hamiltonian representation
        # They are all hybrid or bosonic `MultiOperatorTerms`
        for multi_terms in cvdv_ops_terms:
            if multi_terms.op_type != "pauli":
                
                # Step1: Remove all multi-pauli hybrid terms and print them separately
                multi_pauli_hybrid_terms = []
                new_multi_terms = []
                for single_term in multi_terms.terms:
                    if single_term._is_multi_pauli_hybrid_term():
                        multi_pauli_hybrid_terms.append(single_term)
                    else:
                        new_multi_terms.append(single_term)
                multi_pauli_hybrid_terms = MultiOperatorTerms(terms=multi_pauli_hybrid_terms)
                new_multi_terms = MultiOperatorTerms(terms=new_multi_terms)
                # In this step, multi_pauli_hybrid_terms is a `MultiOperatorTerms` object inside every `SingleOperatorTerm` with multiple pauli operators;
                # new_multi_terms is a `MultiOperatorTerms` object inside every `SingleOperatorTerm` with only one or less pauli operator

                
                # If there are multi-pauli hybrid terms, parse them to intermediate Hamiltonian at first
                if len(multi_pauli_hybrid_terms.terms) > 0:
                    serializable_pauli_hybrid_terms = multi_pauli_hybrid_terms._to_merged_multi_operator_terms()
                    
                    # Convert serializable_pauli_hybrid_terms to intermediate Hamiltonian(str)
                    for serializable_pauli_hybrid_term in serializable_pauli_hybrid_terms:
                        intermediate_hamiltonian = serialize_multi_operator_terms_to_intermediate_hamiltonian(serializable_pauli_hybrid_term, final_pauli_length, self.is_sigma)
                        if self.is_print_debug_comments: self.output_manager.add_comment("Intermediate Hamiltonian")
                        self.output_manager.add_instruction(intermediate_hamiltonian)
                
                # If there are still multi-operator terms, parse them to intermediate Hamiltonian at last
                if len(new_multi_terms.terms) > 0:
                    serializable_multi_operator_terms = new_multi_terms._to_merged_multi_operator_terms()
                    
                    # Convert serializable_multi_operator_terms to intermediate Hamiltonian(str)
                    for serializable_multi_operator_term in serializable_multi_operator_terms:
                        intermediate_hamiltonian = serialize_multi_operator_terms_to_intermediate_hamiltonian(serializable_multi_operator_term, final_pauli_length, self.is_sigma)
                        if self.is_print_debug_comments: self.output_manager.add_comment("Intermediate Hamiltonian")
                        self.output_manager.add_instruction(intermediate_hamiltonian)
            else:
                # This should never happen as we filter out Pauli terms earlier
                warnings.warn("Exception: Unexpected Pauli term found in cvdv_ops_terms - this indicates a logic error")

        return collected_multi_ops_terms

    # Deprecated
    # Output function for `pauli` terms
    def _write_pauli_ops_terms(self, pauli_ops_terms: MultiOperatorTerms, final_length: int, is_sigma: bool = False) -> None:
        """
        Convert and write Pauli operator terms to fixed-length string representations.
        
        Args:
            pauli_ops_terms: Collection of Pauli operator terms to process
            final_length: Target length for the Pauli string representation
            is_sigma: Flag indicating whether sigma terms are present
            
        Raises:
            ValueError: If operator position exceeds the final length
        """
        if self.is_print_debug_comments: self.output_manager.add_comment("Pauli String Representations")
        base_pauli_list = ["I"] * final_length
        
        for single_pauli_term in pauli_ops_terms.terms:
            pauli_list = base_pauli_list.copy()
            
            for op in single_pauli_term.sub_terms:
                pos = 2 * op.index + op.sigma if is_sigma else op.index

                if op.index < 0:
                    raise ValueError(f"Operator index {op.index} is less than 0, please check there are no out of range Pauli operators")
                if op.sigma is not None and op.sigma < 0:
                    raise ValueError(f"Operator sigma {op.sigma} is less than 0, please check there are no out of range Pauli operators")
                if pos >= final_length:
                    raise ValueError(f"Operator position {pos} exceeds maximum length {final_length}")
                    
                if pauli_list[pos] != "I":
                    existing_op = PauliOperator(pauli_list[pos], op.index, op.sigma)
                    pauli_list[pos] = (existing_op * op).operator_type
                else:
                    pauli_list[pos] = op.operator_type
                    
            # get the raw text of the coefficient and wrap it in parentheses
            # multiply the -i to the coefficient
            coef = get_raw_coefficient(single_pauli_term.coefficient, multi_negative_i=True)
            instr = f"pauli({coef}): " + "".join(pauli_list)
            self.output_manager.add_instruction(instr)
        return
    
    # Deprecated
    # Output function for `hybrid` and `bosonic` terms
    def _parse_multi_operator_terms_to_intermediate_hamiltonian(self, multi_operator_terms: MultiOperatorTerms, is_sigma: bool = False) -> str:
        """
        Parse MultiOperatorTerms object into an intermediate Hamiltonian string representation.
        
        This function converts a MultiOperatorTerms object into a string representation that can be used
        for further processing. It handles Pauli, Bosonic, and Fermionic operators with proper error handling.
        
        Args:
            multi_operator_terms (MultiOperatorTerms): The multi-operator terms to parse
            is_sigma (bool, optional): Flag to indicate if the operator is a sigma operator. Defaults to False.
            
        Returns:
            str: The intermediate Hamiltonian string representation
            
        Raises:
            ValueError: If an unsupported operator type is encountered
            TypeError: If the input is not a valid MultiOperatorTerms object
        """
        if not isinstance(multi_operator_terms, MultiOperatorTerms):
            raise TypeError("Input must be a MultiOperatorTerms object")

        def _parse_single_op(op: BaseOperator) -> str:
            """
            Parse a single operator into its string representation.
            
            Args:
                op (BaseOperator): The operator to parse
                
            Returns:
                str: String representation of the operator
                
            Raises:
                ValueError: If operator type is not supported
            """
            if not isinstance(op, BaseOperator):
                raise TypeError(f"Expected BaseOperator, got {type(op)}")

            if isinstance(op, PauliOperator):
                index = op.index if not is_sigma else op.index * 2 + op.sigma
                operator_map = {
                    "I": f"sigma(0, {index})",
                    "X": f"sigma(1, {index})",
                    "Y": f"sigma(2, {index})",
                    "Z": f"sigma(3, {index})"
                }
                if op.operator_type not in operator_map:
                    raise ValueError(f"Unsupported Pauli operator type: {op.operator_type}")
                return operator_map[op.operator_type]
                
            elif isinstance(op, BosonicOperator):
                if op.operator_type not in ["creation", "annihilation"]:
                    raise ValueError(f"Unsupported Bosonic operator type: {op.operator_type}")
                return f"b({op.parameters[0]})" if op.operator_type == "annihilation" else f"dagger(b({op.parameters[0]}))"
                
            elif isinstance(op, FermionicOperator):
                raise NotImplementedError("Fermionic operators are currently not supported")
                
            raise ValueError(f"Unsupported operator type: {type(op)}")
        
        def _parse_single_term(term: SingleOperatorTerm) -> str:
            """
            Parse a single operator term into its string representation.
            
            Args:
                term (SingleOperatorTerm): The term to parse
                
            Returns:
                str: String representation of the term
            """
            
            if not is_only_real_part(term.coefficient):
                warnings.warn(f"Warning: Current we only support real number as numeric coefficient for hybrid and bosonic terms.")
                # multiply the -i to the coefficient
                coef = get_raw_coefficient(term.coefficient, is_only_real_part=True, multi_negative_i=True)
            else:
                # multiply the -i to the coefficient
                coef = get_raw_coefficient(term.coefficient, multi_negative_i=True)
            
            if not term.sub_terms:
                return f"prod(({coef}))"
                
            operators = [_parse_single_op(op) for op in term.sub_terms]
            return f"prod(({coef}),{','.join(operators)})"
        
        def _parse_multi_term(term: MultiOperatorTerms) -> str:
            """
            Parse multiple operator terms into their string representation.
            
            Args:
                term (MultiOperatorTerms): The terms to parse
                
            Returns:
                str: String representation of the terms
            """
            if not term.terms:
                return ""
                
            terms = [_parse_single_term(single_term) for single_term in term.terms]
            if len(terms) == 1:
                return terms[0]
            return f"sum({','.join(terms)})"
        
        op_type = multi_operator_terms.op_type
        if op_type == "pauli" or op_type == "fermionic":
            warnings.warn(f"Exception: {op_type} terms are not supported in intermediate Hamiltonian")
        try:
            return f"{op_type}: exp({_parse_multi_term(multi_operator_terms)})"
        except Exception as e:
            raise ValueError(f"Failed to parse multi-operator terms: {str(e)}")

    # Deprecated
    def _parse_multi_pauli_hybrid_terms_to_intermediate_hamiltonian(self, multi_pauli_hybrid_terms: MultiOperatorTerms, final_pauli_length: int, is_sigma: bool = False) -> str:
        """
        Parse multi-pauli hybrid terms into their string representation.
        """
        
        def _parse_multi_pauli_single_term_to_PAULI_STRING(multi_pauli_single_term: SingleOperatorTerm, final_pauli_length: int, is_sigma: bool = False) -> str:
            """
            Parse a single pauli term into its string representation.
            """
            if multi_pauli_single_term.op_type != "pauli":
                raise ValueError("Input must be a SingleOperatorTerm object with op_type 'pauli'")
            
            pauli_str = "paulistring("
            
            base_pauli_list = ["I"] * final_pauli_length
            
            for op in multi_pauli_single_term.sub_terms:
                pos = 2 * op.index + op.sigma if is_sigma else op.index
                
                if op.index < 0:
                    raise ValueError(f"Operator index {op.index} is less than 0, please check there are no out of range Pauli operators")
                if op.sigma is not None and op.sigma < 0:
                    raise ValueError(f"Operator sigma {op.sigma} is less than 0, please check there are no out of range Pauli operators")
                if pos >= final_pauli_length:
                    raise ValueError(f"Operator position {pos} exceeds maximum length {final_pauli_length}")
                    
                if base_pauli_list[pos] != "I":
                    existing_op = PauliOperator(base_pauli_list[pos], op.index, op.sigma)
                    base_pauli_list[pos] = (existing_op * op).operator_type
                else:
                    base_pauli_list[pos] = op.operator_type
                    
            return pauli_str+"".join(base_pauli_list)+")"
        
        def _parse_bosonic_operator(bosonic_single_term: SingleOperatorTerm) -> str:
            """
            Parse a bosonic operator into its string representation.
            """
            if bosonic_single_term.op_type != "bosonic":
                raise ValueError("Input must be a SingleOperatorTerm object with op_type 'bosonic'")
            
            bosonic_str = ""
            
            for op in bosonic_single_term.sub_terms:
                if op.operator_type not in ["creation", "annihilation"]:
                    raise ValueError(f"Unsupported Bosonic operator type: {op.operator_type}")
                bosonic_str += f"dagger(b({op.parameters[0]}))" if op.operator_type == "annihilation" else f"b({op.parameters[0]})"
                bosonic_str += ","
            
            return bosonic_str[:-1]

        result = "exp(sum("
        if multi_pauli_hybrid_terms.op_type != "hybrid":
            raise ValueError("Input must be a MultiOperatorTerms object with op_type 'hybrid'")
        
        single_3_tuples = multi_pauli_hybrid_terms._split_multi_pauli_hybrid_multi_terms()
        for single_3_tuple in single_3_tuples:
            coefficient, pauli, bosonic = single_3_tuple
            if coefficient.op_type != "coefficient":
                raise ValueError("Coefficient must be a coefficient term")
            if pauli.op_type != "pauli":
                raise ValueError("Pauli must be a pauli term")
            if bosonic.op_type != "bosonic":
                raise ValueError("Bosonic must be a bosonic term")

            # multiply the -i to the coefficient
            raw_coefficient = get_raw_coefficient(coefficient.coefficient, multi_negative_i=True)
            pauli_str = _parse_multi_pauli_single_term_to_PAULI_STRING(pauli, final_pauli_length, is_sigma)
            bosonic_str = _parse_bosonic_operator(bosonic)
            sub_result = f"prod(({raw_coefficient}),{pauli_str},{bosonic_str})"
            
            result += sub_result + ","

        return "hybrid: "+result[:-1]+"))"