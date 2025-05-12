# src.ast_nodes

from typing import List, Optional, Union
from enum import Enum

#
# 1. Enum Definitions
#

class BinaryOperator(Enum):
    """Enumeration for binary operators: +, -, *, /, ^."""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "^"


class AccumulationType(Enum):
    """Enumeration for accumulation operations: Sum_over, Prod_over, TensorProd_over."""
    SUM_OVER = "Sum_over"
    PROD_OVER = "Prod_over"
    TENSORPROD_OVER = "TensorProd_over"


#
# 2. Base AST Node
#

class ASTNode:
    """Base class for all AST nodes, optionally storing line and column information."""
    def __init__(self, line: Optional[int] = None, column: Optional[int] = None):
        self.line = line
        self.column = column


#
# 3. Statement-Level Nodes
#

class ProgramNode(ASTNode):
    """Root node of the program, containing multiple statements."""
    def __init__(self, statements: List['StatementNode'], line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.statements = statements


class StatementNode(ASTNode):
    """Base class for all statement nodes."""
    pass


class ConstDeclarationNode(StatementNode):
    """Represents: Const <name> = <value_expr>;"""
    def __init__(self, name: str, value_expr: 'ExpressionNode', line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.name = name
        self.value_expr = value_expr


class RangeDeclarationNode(StatementNode):
    """Represents: Range <name> = [start_expr, end_expr, step_expr];"""
    def __init__(self, name: str, start_expr: 'ExpressionNode', end_expr: 'ExpressionNode',
                 step_expr: 'ExpressionNode', line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.name = name
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.step_expr = step_expr


class AssignmentNode(StatementNode):
    """Represents: <identifier> = <expression>;"""
    def __init__(self, identifier: str, expression: 'ExpressionNode', line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.identifier = identifier
        self.expression = expression


class ExpressionStatementNode(StatementNode):
    """Represents a standalone expression statement: <expression>;"""
    def __init__(self, expression: 'ExpressionNode', line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.expression = expression


#
# 4. Expression-Level Nodes
#

class ExpressionNode(ASTNode):
    """Base class for all expression nodes."""
    pass


class BinaryOpNode(ExpressionNode):
    """Represents a binary operation: expr1 + expr2, expr1 - expr2, etc."""
    def __init__(self, op: BinaryOperator, left: 'ExpressionNode', right: 'ExpressionNode',
                 line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.op = op
        self.left = left
        self.right = right


class ParenExpressionNode(ExpressionNode):
    """Represents a parenthesized expression: (expr)."""
    def __init__(self, expression: 'ExpressionNode', line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.expression = expression


class UnaryMinusNode(ExpressionNode):
    """Represents a unary negation: -expr."""
    def __init__(self, expression: 'ExpressionNode', line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.expression = expression


class TensorProdNode(ExpressionNode):
    """Represents: TensorProd(expr1, expr2, ...)."""
    def __init__(self, factor_exprs: List[ExpressionNode], line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.factor_exprs = factor_exprs


class AccumulationNode(ExpressionNode):
    """
    Represents:
      Sum_over(...) { ... }
      Prod_over(...) { ... }
      TensorProd_over(...) { ... }
    """
    def __init__(self, op: AccumulationType, range_vars: List['RangeVarNode'], body_expr: ExpressionNode,
                 line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.op = op
        self.range_vars = range_vars
        self.body_expr = body_expr


class RangeVarNode(ASTNode):
    """
    Represents a range variable:
      - i  (simple identifier)
      - Range i = [start, end, step] (inline range declaration)
    """
    def __init__(self, var_name: str, start_expr: Optional[ExpressionNode], end_expr: Optional[ExpressionNode],
                 step_expr: Optional[ExpressionNode], is_inline: bool = False,
                 line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.var_name = var_name
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.step_expr = step_expr
        self.is_inline = is_inline


class QuantumOpNode(ExpressionNode):
    """
    Represents quantum operators:
      FC[i][sigma], FA[i], FN[i], BC[i], BA[i], Pauli_X[i], etc.
    """
    def __init__(self, op_type: str, indices: List[ExpressionNode], line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.op_type = op_type
        self.indices = indices


class ImagNode(ExpressionNode):
    """Represents the imaginary unit: imag."""
    def __init__(self, line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)


class NumberLiteralNode(ExpressionNode):
    """Represents a numeric literal."""
    def __init__(self, value: Union[int, float], line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.value = value


class IdentifierNode(ExpressionNode):
    """Represents an identifier (variable reference)."""
    def __init__(self, name: str, line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.name = name


class ExpressionListNode(ASTNode):
    """
    Represents a comma-separated list of expressions: (expr, expr, expr, ...).
    This node may be unnecessary if expressions are directly stored in other nodes.
    """
    def __init__(self, expressions: List[ExpressionNode], line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(line, column)
        self.expressions = expressions
