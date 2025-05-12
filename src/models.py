# src.models

from typing import List, Dict, Tuple, Literal
from collections import defaultdict
from copy import deepcopy
import warnings


class BaseOperator:
    """Base class for all quantum operators."""
    def __init__(self, operator_type: str, *parameters: int):
        """
        Initialize a generic quantum operator.

        :param operator_type: Type of operator (e.g., 'bosonic', 'pauli', 'fermionic_number', 'creation', 'annihilation').
        :param parameters: Arbitrary integer parameters related to the operator.
        """
        self.operator_type = operator_type
        self.parameters: List[int] = list(parameters)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.operator_type}, parameters={self.parameters})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseOperator):
            return NotImplemented
        return self.operator_type == other.operator_type and self.parameters == other.parameters
    

class PauliOperator(BaseOperator):
    """Class representing a Pauli operator with types X, Y, Z, and I."""
    VALID_TYPES = {'X', 'Y', 'Z', 'I'}
    
    # Static lookup table for Pauli multiplication
    MULTIPLICATION_TABLE: Dict[Tuple[str, str], str] = {
        ("X", "X"): "I", ("Y", "Y"): "I", ("Z", "Z"): "I",
        ("X", "Y"): "Z", ("Y", "X"): "Z",
        ("Y", "Z"): "X", ("Z", "Y"): "X",
        ("Z", "X"): "Y", ("X", "Z"): "Y",
    }

    def __init__(self, pauli_type: str, index: int, sigma: int):
        if pauli_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid Pauli type '{pauli_type}'. Must be one of {self.VALID_TYPES}.")
        super().__init__(pauli_type, index, sigma)
        self.op_type = "pauli"
        self.index, self.sigma = index, sigma
    
    def __mul__(self, other: 'PauliOperator') -> 'PauliOperator':
        """
        Multiply two Pauli operators acting on the same site.
        Ignores phase factors and coefficients.
        """
        if (self.index, self.sigma) != (other.index, other.sigma):
            raise ValueError("Pauli operators act on different sites and cannot be multiplied directly.")

        if self.operator_type == "I":
            return other
        if other.operator_type == "I":
            return self

        result_type = self.MULTIPLICATION_TABLE.get((self.operator_type, other.operator_type))
        if result_type is None:
            raise ValueError(f"Invalid multiplication of Pauli operators: {self.operator_type} and {other.operator_type}")

        return PauliOperator(result_type, self.index, self.sigma)
    
    
class PauliString:
    """Class representing a list of Pauli operators forming a Pauli string."""
    def __init__(self, operator_list: List[PauliOperator]):
        self.operator_list = operator_list
    
    def __str__(self) -> str:
        """Convert the list of Pauli operators to a string representation."""
        if not self.operator_list:
            return "I"  # Default case: Identity Pauli string
        max_index = max(op.index for op in self.operator_list)
        length = (max_index + 1) * 2  # Compute final length
        pauli_string = ["I"] * length  

        for operator in self.operator_list:
            replace_index = 2 * operator.index + operator.sigma
            if replace_index >= len(pauli_string):
                raise ValueError(f"Index {replace_index} out of bounds for Pauli string length {length}")
            if pauli_string[replace_index] != "I":
                raise ValueError(f"Conflict: Pauli string already contains {pauli_string[replace_index]} at index {replace_index}")
            pauli_string[replace_index] = operator.operator_type
        
        return "".join(pauli_string)

    def __mul__(self, other: 'PauliString') -> 'PauliString':
        """
        Multiply two PauliString objects.
        For operators acting on the same site, apply the Pauli multiplication rule.
        For disjoint sites, simply combine the operator lists.
        """
        op_dict = { (op.index, op.sigma): op for op in self.operator_list }

        for op in other.operator_list:
            key = (op.index, op.sigma)
            if key in op_dict:
                op_dict[key] = op_dict[key] * op  # Use __mul__ from PauliOperator
            else:
                op_dict[key] = op
        
        return PauliString(list(op_dict.values()))

    def tensor_product(self, other: 'PauliString') -> 'PauliString':
        """
        Compute the tensor product of two PauliString objects.
        This merges their operator lists without applying multiplication.
        """
        op_dict = { (op.index, op.sigma): op for op in self.operator_list }

        for op in other.operator_list:
            key = (op.index, op.sigma)
            if key in op_dict and op_dict[key].operator_type != op.operator_type:
                raise ValueError(f"Conflict in tensor product at index={op.index}, sigma={op.sigma}: "
                                f"{op_dict[key].operator_type} vs {op.operator_type}")
            op_dict[key] = op

        return PauliString(list(op_dict.values()))
    

class QuantumOperator(BaseOperator):
    """Base class for quantum operators (fermionic and bosonic)."""
    def __init__(self, is_creation: bool, *parameters: int):
        operator_type = "creation" if is_creation else "annihilation"
        super().__init__(operator_type, *parameters)

class FermionicOperator(QuantumOperator):
    """Class representing a fermionic operator."""
    # Note: The encoding logic is moved to encoding.py.
    pass

class FermionicNumber(BaseOperator):
    """Class representing a fermionic number operator."""
    def __init__(self, *parameters: int):
        """
        Initialize the fermionic number operator.
        
        :param parameters: Arbitrary integer parameters related to the fermionic number.
        """
        super().__init__("fermionic_number", *parameters)
    # Note: The encoding logic is moved to encoding.py.


class BosonicOperator(QuantumOperator):
    """Class representing a bosonic operator."""
    pass

# Deprecated
class BosonicOperatorSum:
    """Class representing a sum of bosonic operators."""
    def __init__(self, operators: List[BosonicOperator]):
        self.operators = operators

    def __repr__(self) -> str:
        string = ""
        # Right now all operators are added, doesn't account for subtraction
        for op in self.operators:
            string += f"{op} + " 
        return f"{self.__class__.__name__}({string[:-3]})"


# Deprecated
# Helper class to encapsulate multiple terms (resulting from addition/subtraction)
class PauliSum:
    def __init__(self, terms):
        """
        :param terms: A list of PauliString objects.
        """
        self.terms = terms
        

class SingleOperatorTerm:
    """
    Class representing a single term in the sum_over node.
    """
    def __init__(
        self,
        op_type: Literal["fermionic", "bosonic", "pauli", "hybrid", "coefficient"] | None = None,
        pauli_term: List[PauliOperator] | None = None,
        fermionic_term: List[FermionicOperator] | None = None,
        bosonic_term: List[BosonicOperator] | None = None,
        coefficient: complex | int | float = 1 + 0j
    ):
        # convert any real number input to complex
        self.coefficient: complex = complex(coefficient)
        
        self.op_type = op_type
        self.pauli_term = pauli_term
        self.fermionic_term = fermionic_term
        self.bosonic_term = bosonic_term
        self.sub_terms = self._initialize_sub_terms()
        self.is_pauli_tensor_bosonic = False
        self.bosonic_coefficient = None
        self.bosonic_term = None
        #self._update_pauli_bosonic_status()

    def __str__(self) -> str:
        """Convert the SingleOperatorTerm object to a string representation."""
        return f"{self.__class__.__name__}({self.op_type}) {self.coefficient} * {self.sub_terms}"

    def _initialize_sub_terms(self) -> List:
        """Initialize sub_terms list based on operator type."""
        if self.op_type is None:
            return []
        if self.op_type == "coefficient":
            return []
            
        sub_terms = []
        if self.op_type == "pauli" and self.pauli_term is not None:
            sub_terms.extend(self.pauli_term)
        elif self.op_type == "fermionic" and self.fermionic_term is not None:
            sub_terms.extend(self.fermionic_term)
        elif self.op_type == "bosonic" and self.bosonic_term is not None:
            sub_terms.extend(self.bosonic_term)
        elif self.op_type == "hybrid":
            if self.pauli_term is not None:
                sub_terms.extend(self.pauli_term)
            if self.fermionic_term is not None:
                sub_terms.extend(self.fermionic_term)
            if self.bosonic_term is not None:
                sub_terms.extend(self.bosonic_term)
        return sub_terms
        
    def __mul__(self, other: 'SingleOperatorTerm') -> 'SingleOperatorTerm':
        result = deepcopy(self)      
        result.coefficient *= other.coefficient     
        if self.op_type == "coefficient" and other.op_type == "coefficient":
            result.op_type = "coefficient"
        elif self.op_type == "coefficient":
            result.op_type = other.op_type
        elif other.op_type == "coefficient":
            result.op_type = self.op_type
        else:
            result.op_type = (
                result.op_type if result.op_type == other.op_type else 'hybrid'
            )
        result.sub_terms.extend(other.sub_terms)
        return result
    
    def __add__(self, other: 'SingleOperatorTerm') -> 'MultiOperatorTerms':
        multi_terms = MultiOperatorTerms()
        if self.op_type == "coefficient" and other.op_type == "coefficient":
            multi_terms.op_type = "coefficient"
            term = deepcopy(self)
            term.coefficient += other.coefficient
            multi_terms.terms.append(term)
            return multi_terms
        elif self.op_type == "coefficient":
            multi_terms.op_type = other.op_type
        elif other.op_type == "coefficient":
            multi_terms.op_type = self.op_type
        else:
            multi_terms.op_type = (
                self.op_type if self.op_type == other.op_type else 'hybrid'
            )
            
        multi_terms.terms.append(self)
        multi_terms.terms.append(other)
        return multi_terms

    def __eq__(self, other: 'SingleOperatorTerm') -> bool:
        if self.op_type != other.op_type or self.coefficient != other.coefficient:
            return False
        own_sub_terms = self._sort_sub_terms(self.sub_terms)
        other_sub_terms = self._sort_sub_terms(other.sub_terms)
        if len(own_sub_terms) != len(other_sub_terms): return False
        for own_sub_term, other_sub_term in zip(own_sub_terms, other_sub_terms):
            if own_sub_term != other_sub_term:
                return False
        return True
    
    def _bosonic_eq_in_parameters(self, other: 'SingleOperatorTerm') -> bool:
        """
        Compare if two SingleOperatorTerm objects have the same bosonic operators
        in terms of their parameters, regardless of order.

        Args:
            other: Another SingleOperatorTerm to compare with

        Returns:
            bool: True if both terms have the same bosonic operators with the same parameters,
                  False otherwise
        """
        # currently only consider bosonic operators
        if self.op_type != "bosonic" or other.op_type != "bosonic":
            return False
            
        own_bit_map = defaultdict(int)
        other_bit_map = defaultdict(int)
        
        for op in self.sub_terms:
            if isinstance(op, BosonicOperator):
                own_bit_map[tuple(op.parameters)] += 1
            else:
                raise ValueError(f"Unexpected subterm type: {type(op)} in _bosonic_eq_in_parameters")
                
        for op in other.sub_terms:
            if isinstance(op, BosonicOperator):
                other_bit_map[tuple(op.parameters)] += 1
            else:
                raise ValueError(f"Unexpected subterm type: {type(op)} in _bosonic_eq_in_parameters")
                
        return own_bit_map == other_bit_map
    
    def _is_multi_pauli_hybrid_term(self) -> bool:
        if self.op_type != "hybrid":
            return False
        if self.pauli_term is not None:
            return len(self.pauli_term) > 1
        count = 0
        for op in self.sub_terms:
            if isinstance(op, PauliOperator):
                count += 1
        return count > 1
    
    def _split_multi_pauli_hybrid_single_term(self) -> Tuple['SingleOperatorTerm', 'SingleOperatorTerm', 'SingleOperatorTerm']:
        """
        Split multi-pauli hybrid single term into a coefficient single term, a pauli single term and a bosonic single term.
        """
        coefficient_term = SingleOperatorTerm(op_type="coefficient", coefficient=self.coefficient)
        pauli_term = []
        bosonic_term = []
        for term in self.sub_terms:
            if isinstance(term, PauliOperator):
                pauli_term.append(term)
            elif isinstance(term, BosonicOperator):
                bosonic_term.append(term)
        pauli_single_term = SingleOperatorTerm(op_type="pauli", pauli_term=pauli_term)
        bosonic_single_term = SingleOperatorTerm(op_type="bosonic", bosonic_term=bosonic_term)
        return coefficient_term, pauli_single_term, bosonic_single_term

    def _is_pauli_tensor_bosonic(self) -> bool:
        """
        Check if this single operator term is a pauli tensor bosonic or only bosonic
        """
        if self.op_type == "coefficient" or self.op_type == "fermionic" or self.op_type == "pauli":
            return False
        if self.fermionic_term != None:
            return False
        
        return True
    
    def _get_bosonic_coefficient(self) -> 'SingleOperatorTerm':
        if not self.is_pauli_tensor_bosonic:
            return None
        subterm_coefficient = self.coefficient
        subterm_pauli_term = []
        
        for term in self.sub_terms:
            if isinstance(term, PauliOperator):
                subterm_pauli_term.append(term)
            elif isinstance(term, BosonicOperator):
                continue
            else:
                raise ValueError(f"Unexpected subterm type: {type(term)} in _get_bosonic_coefficient")
                
        if len(subterm_pauli_term) == 0:
            return SingleOperatorTerm(op_type="coefficient", coefficient=subterm_coefficient)
        return SingleOperatorTerm(op_type="pauli", pauli_term=self._sort_sub_terms(subterm_pauli_term), coefficient=subterm_coefficient)
    
    def _get_bosonic_term(self) -> 'SingleOperatorTerm':
        if not self.is_pauli_tensor_bosonic:
            return None
        subterm_bosonic_term = []
        
        for term in self.sub_terms:
            if isinstance(term, BosonicOperator):
                subterm_bosonic_term.append(term)
            elif isinstance(term, PauliOperator):
                continue
            else:
                raise ValueError(f"Unexpected subterm type: {type(term)} in _get_bosonic_term")
                
        return SingleOperatorTerm(op_type="bosonic", bosonic_term=self._sort_sub_terms(subterm_bosonic_term))

    def _update_pauli_bosonic_status(self) -> None:
        """
        Update the pauli_tensor_bosonic status of the single operator term.
        """
        self.is_pauli_tensor_bosonic = self._is_pauli_tensor_bosonic()
        self.bosonic_coefficient = self._get_bosonic_coefficient()
        self.bosonic_term = self._get_bosonic_term()
    
    @staticmethod
    def _sort_sub_terms(sub_terms: List[BaseOperator]) -> List[BaseOperator]:
        """
        Sort the sub-terms of a single operator term according to the following rules:
        1. Pauli operators are placed first
        2. Bosonic operators are placed last
        3. Fermionic operators are ignored (with a warning)
        4. For operators of the same type, sort by their parameters:
           - Pauli operators: sort by [index, sigma] parameters
           - Bosonic operators: sort by default order

        Args:
            sub_terms: List of BaseOperator objects to be sorted

        Returns:
            List[BaseOperator]: Sorted list of operators, excluding any fermionic operators

        Note:
            This function will emit a warning if any fermionic operators are encountered
            in the input list.
        """
        pauli_ops = []
        bosonic_ops = []
        
        for op in sub_terms:
            if isinstance(op, PauliOperator):
                pauli_ops.append(op)
            elif isinstance(op, BosonicOperator):
                bosonic_ops.append(op)
            elif isinstance(op, FermionicOperator) or isinstance(op, FermionicNumber):
                warnings.warn(f"Ignoring fermionic operator in sub-terms sorting: {op}")
            else:
                raise ValueError(f"Unexpected operator type in sub-terms: {type(op)}")

        # Sort Pauli operators by [index, sigma] parameters
        pauli_ops.sort(key=lambda x: (x.index, x.sigma))
        
        return pauli_ops + bosonic_ops


class MultiOperatorTerms:
    """
    Class representing the ops terms in the sum_over node.
    """
    def __init__(self, terms: List[SingleOperatorTerm] | None = None):
        self.terms: List[SingleOperatorTerm] = terms or []
        self.op_type = self._determine_op_type()

    def __str__(self) -> str:
        """Convert the MultiOperatorTerms object to a string representation."""
        number_of_terms = len(self.terms)
        if number_of_terms == 0:
            return f"{self.__class__.__name__}({self.op_type}) (0 terms) "
        string = ""
        string += f"{self.__class__.__name__}({self.op_type}) ({number_of_terms} terms) "
        for term in self.terms:
            string += f"{term} + "
        return string[:-3]

    def _determine_op_type(self) -> str | None:
        """Determine the operator type based on the terms."""
        if not self.terms:
            return None
        
        if len(self.terms)<=1:
            return self.terms[0].op_type
        
        first_type = None
        for term in self.terms:
            if term.op_type == "coefficient":
                continue
            if first_type is None:
                first_type = term.op_type
            elif first_type != term.op_type:
                return "hybrid"
            
        if first_type is None:
            return "coefficient"
        return first_type

    def __mul__(self, other: 'MultiOperatorTerms') -> 'MultiOperatorTerms':
        temp_terms = []
        for l_term in self.terms:
            for r_term in other.terms:
                temp_terms.append(l_term * r_term)
        return MultiOperatorTerms(temp_terms)

    def __add__(self, other: 'MultiOperatorTerms') -> 'MultiOperatorTerms':
        """
        Add two MultiOperatorTerms objects.
        """
        self.terms.extend(other.terms)
        self.op_type = self._determine_op_type()
        return self
    
    def _remove_pauli_single_terms(self) -> Tuple['MultiOperatorTerms', List['MultiOperatorTerms']]:
        """
        Remove pauli single terms from the MultiOperatorTerms object, return this remaining MultiOperatorTerms object.
        Construct a new MultiOperatorTerms object with every removed pauli single term, return a list of these removed pauli single terms MultiOperatorTerms object.
        """
        remaining_terms = []
        removed_terms = []
        for term in self.terms:
            if term.op_type == "pauli":
                removed_terms.append(MultiOperatorTerms([term]))
            else:
                remaining_terms.append(term)

        return MultiOperatorTerms(remaining_terms), removed_terms
    
    def _largest_pauli_index(self) -> int:
        """
        Return the largest Pauli index in the MultiOperatorTerms object.
        """
        max_index = 0
        for term in self.terms:
            if term.op_type == "pauli":
                for op in term.sub_terms:
                    max_index = max(max_index, op.index)
            else:
                for op in term.sub_terms:
                    if isinstance(op, PauliOperator):
                        max_index = max(max_index, op.index)
        
        return max_index
        
    def _is_pauli_term_has_sigma(self) -> bool:
        """
        Check if the Pauli term has sigma.
        """
        for term in self.terms:
            if term.op_type == "pauli":
                if term.sub_terms is None:
                    warnings.warn(f"Exception: Pauli term is None: {term}")
                for op in term.sub_terms:
                    if op.sigma is not None and op.sigma > 0:
                        return True
            else:
                for op in term.sub_terms:
                    if isinstance(op, PauliOperator):
                        if op.sigma is not None and op.sigma > 0:
                            return True
        return False
        
    def _simplify(self) -> 'MultiOperatorTerms':
        """
        Combine terms that have identical operator content
        (same op_type and same sub_terms not necessarily in the same order) by summing
        their complex coefficients. Terms whose total coefficient
        becomes 0 are removed.
        """
        # key: (op_type, frozenset(multiset of sub_term representations))
        buckets: dict[tuple[str, frozenset[str]], complex] = defaultdict(complex)
        samples: dict[tuple[str, frozenset[str]], SingleOperatorTerm] = {}
        new_terms: list[SingleOperatorTerm] = []
        
        for term in self.terms:
            # Currently we don't simplify pauli terms and coefficient terms
            if term.op_type in ("pauli", "coefficient"):
                new_terms.append(term)
                continue
            
            # Using repr(op) as a hashable surrogate; replace with a stronger
            sequence_key = tuple(repr(op) for op in term.sub_terms)
            key = (term.op_type, sequence_key)
                
            buckets[key] += term.coefficient
            # keep one exemplar term so we can clone it later
            if key not in samples:
                samples[key] = term

        for key, coeff in buckets.items():
            # if abs(coeff) < 1e-12:      # skip zero‑weight terms
            #     continue
            term_proto = deepcopy(samples[key])
            term_proto.coefficient = coeff      # updated coefficient
            new_terms.append(term_proto)

        return MultiOperatorTerms(new_terms)

    def _split_multi_pauli_hybrid_multi_terms(self) -> List[Tuple[SingleOperatorTerm, SingleOperatorTerm, SingleOperatorTerm]]:
        """
        Split multi-pauli hybrid multi-terms into multiple 3-tuple of (coefficient, pauli, bosonic) single terms.
        """
        result = []
        for single_term in self.terms:
            if single_term.op_type == "hybrid":
                coefficient, pauli, bosonic = single_term._split_multi_pauli_hybrid_single_term()
                result.append((coefficient, pauli, bosonic))
                
        return result
    
    def _to_merged_multi_operator_terms(self) -> List['SerializableMultiOperatorTerms']:
        """
        Convert the MultiOperatorTerms object to a list of SerializableMultiOperatorTerms objects.
        """
        result = []
        single_terms = deepcopy(self.terms)
        coefficient_dict = {}
        
        for single_term in single_terms:
            single_term._update_pauli_bosonic_status()
            
            if not single_term.is_pauli_tensor_bosonic:
                warnings.warn(f"Exception: A non-pauli tensor bosonic term is found in _to_merged_multi_operator_terms: {single_term}")
                continue

            # Get the bosonic coefficient and term
            bosonic_coefficient = single_term.bosonic_coefficient
            bosonic_term = single_term.bosonic_term

            # Create a key for the coefficient dictionary using the string representation of bosonic_coefficient
            coeff_key = str(bosonic_coefficient)
            
            if coeff_key not in coefficient_dict:
                coefficient_dict[coeff_key] = {
                    'coefficient_pauli_term': bosonic_coefficient,
                    'bosonic_terms': []
                }
            
            # Check if this bosonic_term is already in the list
            found = False
            for existing_term_list in coefficient_dict[coeff_key]['bosonic_terms']:
                if existing_term_list[0]._bosonic_eq_in_parameters(bosonic_term):
                    found = True
                    existing_term_list.append(bosonic_term)
            
            if not found:
                coefficient_dict[coeff_key]['bosonic_terms'].append([bosonic_term])
        
        # Convert the dictionary to SerializableMultiOperatorTerms objects
        for coeff_data in coefficient_dict.values():
            if coeff_data['bosonic_terms']:
                for bosonic_term_list in coeff_data['bosonic_terms']:
                    merged_term = SerializableMultiOperatorTerms(
                        coefficient_pauli_term=coeff_data['coefficient_pauli_term'],
                        bosonic_terms=bosonic_term_list
                    )
                    result.append(merged_term)
        
        return result
        
    
class SerializableMultiOperatorTerms:
    """
    Class representing the merged multi-operator terms, which contains two SingleOperatorTerm objects, first is only one term, which is coefficient and pauli operator sequence, second is a List of bosonic operator sequence, which can allow the merged bosonic operator sequence to be tensor producted with same pauli operator coefficient sequence.
    """
    def __init__(self, coefficient_pauli_term: SingleOperatorTerm, bosonic_terms: List[SingleOperatorTerm]):
        self.coefficient_pauli_term = coefficient_pauli_term
        self.bosonic_terms = bosonic_terms
        self.op_type = self._determine_op_type() # pauli, bosonic, hybrid, coefficient
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.op_type}) ({self.coefficient_pauli_term}) * [{' + '.join([str(term) for term in self.bosonic_terms])}] "
    
    def _determine_op_type(self) -> str:
        """
        Determine the operator type based on the terms.
        """
        if self.coefficient_pauli_term.op_type == "pauli" and self.bosonic_terms[0].op_type == "bosonic":
            return "hybrid"
        elif self.coefficient_pauli_term.op_type == "pauli" and self.bosonic_terms[0].op_type == "coefficient":
            return "pauli"
        elif self.coefficient_pauli_term.op_type == "coefficient" and self.bosonic_terms[0].op_type == "bosonic":
            return "bosonic"
        else:
            return "coefficient"
        