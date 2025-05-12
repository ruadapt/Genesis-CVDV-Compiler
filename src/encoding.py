# src/encoding.py

from typing import List, Tuple
from src.models import FermionicOperator, FermionicNumber, PauliOperator, PauliString
from src.ast_nodes import *

def transform_QuantumOpNode_jw_encoding(ast_root: ASTNode) -> None:
    """
    Transform the AST by replacing QuantumOpNode nodes which represent
    a FermionicOperator or FermionicNumber into Pauli operator nodes
    using the Jordan-Wigner encoding.

    The transformations are as follows:
        c_{j,\sigma}^\dagger -> (1/2) * (X_{j,\sigma} - i Y_{j,\sigma}) ⊗ TensorProd_over(Range k = [0, j, 1]){ Pauli_Z[k][\sigma] }
        c_{j,\sigma}     -> (1/2) * (X_{j,\sigma} + i Y_{j,\sigma}) ⊗ TensorProd_over(Range k = [0, j, 1]){ Pauli_Z[k][\sigma] }
        n_{j,\sigma}    -> (1/2) * (I_{j,\sigma} - Z_{j,\sigma})
    """
    if ast_root is None:
        return

    # If the current node is a QuantumOpNode and its op_type is one of FC, FA, or FN, then transform it.
    if isinstance(ast_root, QuantumOpNode) and ast_root.op_type in {"FC", "FA", "FN"}:
        # Check that the indices list has at least two elements: j and sigma
        if len(ast_root.indices) < 2:
            raise ValueError("QuantumOpNode does not have enough indices for JW transformation.")
        
        j_expr = ast_root.indices[0]
        sigma_expr = ast_root.indices[1]
        
        # Construct the basic Pauli operator nodes (using the same j and sigma parameters).
        pauli_x = QuantumOpNode("Pauli_X", [j_expr, sigma_expr],
                                line=ast_root.line, column=ast_root.column)
        pauli_y = QuantumOpNode("Pauli_Y", [j_expr, sigma_expr],
                                line=ast_root.line, column=ast_root.column)
        pauli_z = QuantumOpNode("Pauli_Z", [j_expr, sigma_expr],
                                line=ast_root.line, column=ast_root.column)
        pauli_i = QuantumOpNode("Pauli_I", [j_expr, sigma_expr],
                                line=ast_root.line, column=ast_root.column)
        
        # Build the operator combination according to op_type.
        if ast_root.op_type == "FC":  # creation operator: c^\dagger
            # Construct (X - iY).
            imag_y = BinaryOpNode(
                op=BinaryOperator.MUL,
                left=ImagNode(line=ast_root.line, column=ast_root.column),
                right=pauli_y,
                line=ast_root.line, column=ast_root.column
            )
            operator_part = BinaryOpNode(
                op=BinaryOperator.SUB,
                left=pauli_x,
                right=imag_y,
                line=ast_root.line, column=ast_root.column
            )
        elif ast_root.op_type == "FA":  # annihilation operator: c
            # Construct (X + iY).
            imag_y = BinaryOpNode(
                op=BinaryOperator.MUL,
                left=ImagNode(line=ast_root.line, column=ast_root.column),
                right=pauli_y,
                line=ast_root.line, column=ast_root.column
            )
            operator_part = BinaryOpNode(
                op=BinaryOperator.ADD,
                left=pauli_x,
                right=imag_y,
                line=ast_root.line, column=ast_root.column
            )
        elif ast_root.op_type == "FN":  # number operator: n
            # Construct (I - Z).
            operator_part = BinaryOpNode(
                op=BinaryOperator.SUB,
                left=pauli_i,
                right=pauli_z,
                line=ast_root.line, column=ast_root.column
            )
        else:
            # Should not reach here.
            operator_part = ast_root

        # Multiply by the constant 1/2.
        half = NumberLiteralNode(0.5, line=ast_root.line, column=ast_root.column)
        transformed_expr = BinaryOpNode(
            op=BinaryOperator.MUL,
            left=half,
            right=operator_part,
            line=ast_root.line, column=ast_root.column
        )
        
        # For creation and annihilation operators, also multiply by the tensor product of Z operators for k < j.
        # Here we use an inline range statement to construct a TensorProd_over node.
        if ast_root.op_type in {"FC", "FA"}:
            # Build the inline range: Range k = [0, j, 1].
            range_var = RangeVarNode(
                var_name="k",
                start_expr=NumberLiteralNode(0, line=ast_root.line, column=ast_root.column),
                end_expr=j_expr, 
                step_expr=NumberLiteralNode(1, line=ast_root.line, column=ast_root.column),
                is_inline=True,
                line=ast_root.line, column=ast_root.column
            )
            # Build the body of the accumulation expression: Pauli_Z[k][sigma].
            k_identifier = IdentifierNode("k", line=ast_root.line, column=ast_root.column)
            pauli_z_k = QuantumOpNode("Pauli_Z", [k_identifier, sigma_expr],
                                      line=ast_root.line, column=ast_root.column)
            # Use an AccumulationNode to represent the TensorProd_over operation.
            z_tensor = AccumulationNode(
                op=AccumulationType.TENSORPROD_OVER,
                range_vars=[range_var],
                body_expr=pauli_z_k,
                line=ast_root.line, column=ast_root.column
            )
            # Combine the transformed expression with the Z operator accumulation result 
            # via a tensor product.
            transformed_expr = TensorProdNode(
                factor_exprs=[transformed_expr, z_tensor],
                line=ast_root.line,
                column=ast_root.column
            )
        
        # Replace the original node with the transformed expression.
        ast_root.__class__ = transformed_expr.__class__
        ast_root.__dict__ = transformed_expr.__dict__

    # Recursively transform child nodes.
    for attr in dir(ast_root):
        if attr.startswith("_") or attr in ("line", "column"):
            continue
        child = getattr(ast_root, attr)
        if isinstance(child, ASTNode):
            transform_QuantumOpNode_jw_encoding(child)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, ASTNode):
                    transform_QuantumOpNode_jw_encoding(item)


#Depreciated
def jw_encoding(fermionic_ops: List[object]) -> List[Tuple[PauliString, PauliString]]:
    """
    Convert a list of fermionic operators (FermionicOperator or FermionicNumber) 
    to a list of PauliString pairs using the Jordan-Wigner encoding.
    
    For FermionicOperator:
      - For an annihilation operator: c_j -> (X_j + iY_j)/2 with preceding Z operators on all sites with indices < j.
      - For a creation operator: c_j† -> (X_j - iY_j)/2 with preceding Z operators on all sites with indices < j.
      
    For FermionicNumber:
      - The encoding is: n_{j,\sigma} -> (I_{j,\sigma} - Z_{j,\sigma})/2.
      
    This function returns a list of tuples, where each tuple contains two PauliString objects representing
    the encoded operator.
    
    :param fermionic_ops: List of fermionic operator objects (FermionicOperator or FermionicNumber).
    :return: List of tuples (PauliString_1, PauliString_2).
    """
    encoded_ops = []
    if not fermionic_ops:
        return encoded_ops
    
    # Determine the maximum indices to calculate a global Pauli string length.
    # Assumes each operator has at least two parameters (index and sigma).
    max_index_0 = max(op.parameters[0] for op in fermionic_ops if len(op.parameters) >= 2)
    max_index_1 = max(op.parameters[1] for op in fermionic_ops if len(op.parameters) >= 2)
    pauli_string_length = (max_index_0 + 1) * (max_index_1 + 1)
    
    for op in fermionic_ops:
        if len(op.parameters) < 2:
            raise ValueError("Fermionic operator must have at least two parameters (index, sigma)")
        index, sigma = op.parameters[0], op.parameters[1]
        
        if isinstance(op, FermionicNumber):
            # Encoding for fermionic number operator: n_{j,sigma} -> (I - Z)/2.
            pauli_list_I = [PauliOperator("I", index, sigma)]
            pauli_list_Z = [PauliOperator("Z", index, sigma)]
            encoded_op = (
                PauliString(pauli_string_length, pauli_list_I),
                PauliString(pauli_string_length, pauli_list_Z)
            )
        elif isinstance(op, FermionicOperator):
            # Encoding for fermionic creation/annihilation operator.
            pauli_list_1 = []
            pauli_list_2 = []
            # For the main operator at site 'index', use X and Y respectively.
            pauli_list_1.append(PauliOperator("X", index, sigma))
            pauli_list_2.append(PauliOperator("Y", index, sigma))
            # Add Z operators for all sites with indices less than 'index'.
            for i in range(index):
                pauli_list_1.append(PauliOperator("Z", i, sigma))
                pauli_list_2.append(PauliOperator("Z", i, sigma))
            encoded_op = (
                PauliString(pauli_string_length, pauli_list_1),
                PauliString(pauli_string_length, pauli_list_2)
            )
        else:
            raise ValueError("Unsupported fermionic operator type")
        
        encoded_ops.append(encoded_op)
    
    return encoded_ops
