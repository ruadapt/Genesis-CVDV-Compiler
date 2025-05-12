# src.serializer

from src.models import *
from src.utils import get_raw_coefficient, is_only_real_part
from typing import List


def serialize_pauli_ops_terms(pauli_ops_terms: MultiOperatorTerms, final_length: int, is_sigma: bool = False) -> List[str]:
    """
    Convert and write Pauli operator terms to fixed-length string representations.
    
    Args:
        pauli_ops_terms: Collection of Pauli operator terms to process
        final_length: Target length for the Pauli string representation
        is_sigma: Flag indicating whether sigma terms are present
        
    Raises:
        ValueError: If operator position exceeds the final length
    """
    serialized_pauli_ops_terms_str = []
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
        serialized_pauli_ops_terms_str.append(instr)
    
    return serialized_pauli_ops_terms_str


def serialize_single_op(op: BaseOperator, is_sigma: bool = False) -> str:
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
        

def serialize_single_op_terms(term: SingleOperatorTerm, is_sigma: bool = False, ignore_coefficient: bool = False, ignore_prod: bool = False) -> str:
    if not is_only_real_part(term.coefficient):
        warnings.warn(f"Warning: Current we only support real number as numeric coefficient for hybrid and bosonic terms.")
        # do not influence the sign of the coefficient
        coef = get_raw_coefficient(term.coefficient, is_only_real_part=True, multi_negative_i=False)
    else:
        # do not influence the sign of the coefficient
        coef = get_raw_coefficient(term.coefficient, multi_negative_i=False)
    
    if not term.sub_terms or term.op_type == "coefficient":
        return f"prod(({coef}))" if not ignore_coefficient else ""
        
    operators = [serialize_single_op(op, is_sigma) for op in term.sub_terms]
    
    if ignore_prod:
        return f"({coef}), {','.join(operators)}" if not ignore_coefficient else f"{','.join(operators)}"
    else:
        return f"prod(({coef}),{','.join(operators)})" if not ignore_coefficient else f"prod({','.join(operators)})"
    

def serialize_multi_op_terms(terms: List[SingleOperatorTerm], is_sigma: bool = False, ignore_coefficient: bool = False, ignore_prod: bool = False) -> str:
    if terms == []:
        return ""

    if len(terms) == 1:
        return serialize_single_op_terms(terms[0], is_sigma, ignore_coefficient, ignore_prod)

    return f"sum({','.join([serialize_single_op_terms(term, is_sigma, ignore_coefficient, ignore_prod=len(term.sub_terms) == 1) for term in terms])})"


def serialize_multi_pauli_operator_term(term: SingleOperatorTerm, final_pauli_length: int, is_sigma: bool = False) -> str:
    """
    Parse multi-pauli term into their string representation.
    """
    if term.op_type != "pauli":
        raise ValueError("Input must be a SingleOperatorTerm object with op_type 'pauli'")    
    pauli_str = "paulistring("
    base_pauli_list = ["I"] * final_pauli_length

    for op in term.sub_terms:
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


def serialize_multi_operator_terms_to_intermediate_hamiltonian(serializable_multi_operator_terms: SerializableMultiOperatorTerms, final_pauli_length: int = 0, is_sigma: bool = False) -> str:
    """
    Serialize multi-operator terms into an intermediate Hamiltonian string representation.
    
    This function converts a SerializableMultiOperatorTerms object into a string representation
    that follows the intermediate Hamiltonian format. It handles different types of terms including
    bosonic coefficients, Pauli terms, and bosonic terms.
    
    Args:
        serializable_multi_operator_terms (SerializableMultiOperatorTerms): The multi-operator terms to serialize
        is_sigma (bool, optional): Flag indicating whether sigma terms are present. Defaults to False.
        
    Returns:
        str: The serialized intermediate Hamiltonian string representation
        
    Raises:
        ValueError: If the operator type is not supported
    """
    # Only take real number coefficient and multiply - i 
    bosonic_coefficient_number = get_raw_coefficient(serializable_multi_operator_terms. coefficient_pauli_term.coefficient, is_only_real_part=True, multi_negative_i=True)
    bosonic_coefficient_pauli_term = SingleOperatorTerm(op_type="pauli", pauli_term=serializable_multi_operator_terms.coefficient_pauli_term.sub_terms)
    bosonic_terms = serializable_multi_operator_terms.bosonic_terms
    op_type = serializable_multi_operator_terms.op_type
    
    # Check for unsupported operator types
    if op_type == "pauli":
        warnings.warn(f"Exception: {op_type} terms are not supported in intermediate Hamiltonian")
    
    # Serialize the bosonic coefficient Pauli term if present
    if len(bosonic_coefficient_pauli_term.sub_terms) == 1:
        bosonic_coefficient_pauli_term_str = serialize_single_op_terms(bosonic_coefficient_pauli_term, is_sigma, ignore_coefficient=True, ignore_prod=True)
    elif len(bosonic_coefficient_pauli_term.sub_terms) < 1:
        bosonic_coefficient_pauli_term_str = ""
    else:
        bosonic_coefficient_pauli_term_str = serialize_multi_pauli_operator_term(bosonic_coefficient_pauli_term, final_pauli_length, is_sigma)
    
    # Serialize the bosonic terms if present
    if len(bosonic_terms) >= 1:
        bosonic_terms_str = serialize_multi_op_terms(bosonic_terms, is_sigma, ignore_coefficient=True, ignore_prod=True)
    elif len(bosonic_terms) < 1:
        bosonic_terms_str = ""
    
    # Start building the result string with the base exponential format
    res = f"{serializable_multi_operator_terms.op_type}: "
    res += f"exp(prod(({bosonic_coefficient_number})"
    
    # Handle different combinations of terms to construct the final string
    if not bosonic_coefficient_pauli_term_str and not bosonic_terms_str:
        # Case 1: Only coefficient number
        return res + "))"
    elif not bosonic_coefficient_pauli_term_str and bosonic_terms_str:
        # Case 2: Coefficient number and bosonic terms
        return res + "," + bosonic_terms_str + "))"
    elif bosonic_coefficient_pauli_term_str and not bosonic_terms_str:
        # Case 3: Coefficient number and Pauli terms (unsupported)
        warnings.warn(f"Exception: {op_type} terms are not supported in intermediate Hamiltonian")
        return res + "," + bosonic_coefficient_pauli_term_str + "))"
    else:
        # Case 4: All components present
        return res + "," + bosonic_coefficient_pauli_term_str + "," + bosonic_terms_str + "))"
