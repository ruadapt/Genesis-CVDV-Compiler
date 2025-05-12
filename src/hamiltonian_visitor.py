# src.hamiltonian_visitor

from antlr4 import *
from generated.hamiltonianDSLVisitor import hamiltonianDSLVisitor
from generated.hamiltonianDSLParser import hamiltonianDSLParser

# Import the AST node classes and enums from ast_nodes.
from src.ast_nodes import (
    ProgramNode, ExpressionStatementNode,
    ConstDeclarationNode, RangeDeclarationNode, AssignmentNode,
    BinaryOpNode, ParenExpressionNode, UnaryMinusNode,
    TensorProdNode, AccumulationNode, RangeVarNode, QuantumOpNode,
    ImagNode, NumberLiteralNode, IdentifierNode, ExpressionListNode,
    BinaryOperator, AccumulationType
)

import logging

# Configure logging:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)


class HamiltonianVisitor(hamiltonianDSLVisitor):
    def __init__(self):
        super().__init__()
        self.symbol_table = {}
        self.result_assignments = [] 

    # ------------------------------------------------
    # program: statementList EOF
    # ------------------------------------------------
    def visitProgram(self, ctx: hamiltonianDSLParser.ProgramContext):
        logger.debug(f"visitProgram at line {ctx.start.line}")
        stmt_list = self.visit(ctx.statementList())
        node = ProgramNode(
            statements=stmt_list,
            line=ctx.start.line,
            column=ctx.start.column
        )
        logger.debug("Created ProgramNode")
        return node

    # ------------------------------------------------
    # statementList: (statement)*
    # ------------------------------------------------
    def visitStatementList(self, ctx: hamiltonianDSLParser.StatementListContext):
        logger.debug("visitStatementList")
        statements = []
        for stmt in ctx.statement():
            stmt_node = self.visit(stmt)
            if stmt_node is not None:
                statements.append(stmt_node)
                logger.debug(f"Added statement node: {stmt_node.__class__.__name__}")
        return statements

    # Helper function to evaluate a “pure” numeric expression (NUMBER, IMAG, unary -, binary + - * / ^) Support Complex Number
    def _evaluate_const_expr(self, expr: ExpressionStatementNode) -> complex:
        """
        Evaluates a “pure” numeric expression (NUMBER, IMAG, unary -, binary + - * / ^)
        into a Python int/float/complex. Raises ValueError otherwise.
        """
        if isinstance(expr, NumberLiteralNode):
            return expr.value
        if isinstance(expr, ImagNode):
            return 1j
        if isinstance(expr, UnaryMinusNode):
            return -self._evaluate_const_expr(expr.expression)
        if isinstance(expr, BinaryOpNode):
            left = self._evaluate_const_expr(expr.left)
            right = self._evaluate_const_expr(expr.right)
            if expr.op == BinaryOperator.ADD:
                return left + right
            if expr.op == BinaryOperator.SUB:
                return left - right
            if expr.op == BinaryOperator.MUL:
                return left * right
            if expr.op == BinaryOperator.DIV:
                return left / right
            if expr.op == BinaryOperator.POW:
                return left ** right
        raise ValueError(f"Expression is not a pure numeric constant: {expr}")

    # ------------------------------------------------
    # statement:
    #   constDeclaration | rangeDeclaration |
    #   IDENTIFIER ASSIGN expression SEMICOLON | expression SEMICOLON
    # ------------------------------------------------
    def visitStatement(self, ctx: hamiltonianDSLParser.StatementContext):
        logger.debug(f"visitStatement at line {ctx.start.line}")
        if ctx.constDeclaration():
            return self.visit(ctx.constDeclaration())
        if ctx.rangeDeclaration():
            return self.visit(ctx.rangeDeclaration())
        if ctx.IDENTIFIER() and ctx.ASSIGN():
            name = ctx.IDENTIFIER().getText()
            expr_node = self.visit(ctx.expression())
            node = AssignmentNode(
                identifier=name,
                expression=expr_node,
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug(f"Created AssignmentNode for identifier '{name}' at line {ctx.start.line}")
            self.result_assignments.append(name)
            self.symbol_table[name] = expr_node
            return node
        else:
            expr_node = self.visit(ctx.expression())
            node = ExpressionStatementNode(
                expression=expr_node,
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug("Created ExpressionStatementNode")
            return node

    # ------------------------------------------------
    # constDeclaration: CONST IDENTIFIER ASSIGN expression SEMICOLON
    # ------------------------------------------------
    def visitConstDeclaration(self, ctx: hamiltonianDSLParser.ConstDeclarationContext):
        logger.debug(f"visitConstDeclaration at line {ctx.start.line}")
        name = ctx.IDENTIFIER().getText()
        value_expr = self.visit(ctx.expression())
        
        # Attempt to fold it to a Python number/complex if possible:
        try:
            numeric_value = self._evaluate_const_expr(value_expr)
            self.symbol_table[name] = numeric_value
            logger.debug(f"Evaluated Const '{name}' to numeric value {numeric_value!r}")
        except ValueError:
            # not a pure numeric expression → keep the AST
            self.symbol_table[name] = value_expr
            logger.debug(f"Stored Const '{name}' as AST (non-numeric)")

        node = ConstDeclarationNode(
            name=name,
            value_expr=value_expr,
            line=ctx.start.line,
            column=ctx.start.column
        )
        return node
    
    
    # ------------------------------------------------
    # rangeDeclaration: RANGE IDENTIFIER ASSIGN LBRACKET expression COMMA expression COMMA expression RBRACKET SEMICOLON
    # ------------------------------------------------
    def visitRangeDeclaration(self, ctx: hamiltonianDSLParser.RangeDeclarationContext):
        logger.debug(f"visitRangeDeclaration at line {ctx.start.line}")
        name = ctx.IDENTIFIER().getText()
        start_expr = self.visit(ctx.expression(0))
        end_expr   = self.visit(ctx.expression(1))
        step_expr  = self.visit(ctx.expression(2))
        node = RangeDeclarationNode(
            name=name,
            start_expr=start_expr,
            end_expr=end_expr,
            step_expr=step_expr,
            line=ctx.start.line,
            column=ctx.start.column
        )
        logger.debug(f"Created RangeDeclarationNode for '{name}'")
        self.symbol_table[name] = (start_expr, end_expr, step_expr)
        return node
        
        # Deprecated
        # Currently, we leave the evaluation of the range parameters to the interpreter.
        
        # Convert the range parameter to a numeric value:
        def get_numeric_value(expr):
            # If the expression is a numeric literal, return its value directly.
            if isinstance(expr, NumberLiteralNode):
                return expr.value
            # If the expression is an identifier, look up its corresponding value in the symbol_table.
            elif isinstance(expr, IdentifierNode):
                value = self.symbol_table.get(expr.name)
                if isinstance(value, NumberLiteralNode):
                    return value.value
                elif isinstance(value, (int, float)):
                    return value
                else:
                    raise ValueError(f"Cannot resolve the numeric value of identifier {expr.name} from the symbol_table")
            else:
                raise ValueError(f"Unsupported type for range parameter: {type(expr)}")

        try:
            start_value = get_numeric_value(start_expr)
            end_value = get_numeric_value(end_expr)
            step_value = get_numeric_value(step_expr)
            self.symbol_table[name] = (start_value, end_value, step_value)
        except Exception as e:
            logger.error(f"Error evaluating range parameters for '{name}': {e}")
            raise

        return node

    # ====================== Expression ======================

    def visitExpression(self, ctx: hamiltonianDSLParser.ExpressionContext):
        return self.visit(ctx.addExpr())

    def visitAddExpr(self, ctx: hamiltonianDSLParser.AddExprContext):
        logger.debug(f"visitAddExpr at line {ctx.start.line}")
        operands = ctx.mulExpr()
        node = self.visit(operands[0])
        for i in range(1, len(operands)):
            op_token = ctx.getChild(2 * i - 1)
            op_text = op_token.getText()
            right = self.visit(operands[i])
            if op_text == '+':
                node = BinaryOpNode(
                    op=BinaryOperator.ADD,
                    left=node,
                    right=right,
                    line=ctx.start.line,
                    column=ctx.start.column
                )
            else:
                node = BinaryOpNode(
                    op=BinaryOperator.SUB,
                    left=node,
                    right=right,
                    line=ctx.start.line,
                    column=ctx.start.column
                )
        logger.debug("Created BinaryOpNode for addExpr")
        return node

    def visitMulExpr(self, ctx: hamiltonianDSLParser.MulExprContext):
        logger.debug(f"visitMulExpr at line {ctx.start.line}")
        operands = ctx.powerExpr()
        node = self.visit(operands[0])
        for i in range(1, len(operands)):
            op_token = ctx.getChild(2 * i - 1)
            op_text = op_token.getText()
            right = self.visit(operands[i])
            if op_text == '*':
                node = BinaryOpNode(
                    op=BinaryOperator.MUL,
                    left=node,
                    right=right,
                    line=ctx.start.line,
                    column=ctx.start.column
                )
            else:
                node = BinaryOpNode(
                    op=BinaryOperator.DIV,
                    left=node,
                    right=right,
                    line=ctx.start.line,
                    column=ctx.start.column
                )
        logger.debug("Created BinaryOpNode for mulExpr")
        return node

    def visitPowerExpr(self, ctx: hamiltonianDSLParser.PowerExprContext):
        logger.debug(f"visitPowerExpr at line {ctx.start.line}")
        if ctx.POWER():
            left = self.visit(ctx.unaryExpr())
            right = self.visit(ctx.powerExpr())
            node = BinaryOpNode(
                op=BinaryOperator.POW,
                left=left,
                right=right,
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug("Created BinaryOpNode for power")
            return node
        else:
            return self.visit(ctx.unaryExpr())

    def visitUnaryExpr(self, ctx: hamiltonianDSLParser.UnaryExprContext):
        logger.debug(f"visitUnaryExpr at line {ctx.start.line}")
        if ctx.MINUS():
            expr = self.visit(ctx.unaryExpr())
            node = UnaryMinusNode(
                expression=expr,
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug("Created UnaryMinusNode")
            return node
        else:
            return self.visit(ctx.primaryExpr())

    # --------------------  primaryExpr --------------------
    # primaryExpr:
    #    LPAREN expression RPAREN
    #  | TENSORPROD LPAREN expressionList RPAREN
    #  | accumulationExpr
    #  | quantumOp
    #  | IMAG
    #  | NUMBER
    #  | IDENTIFIER
    def visitPrimaryExpr(self, ctx: hamiltonianDSLParser.PrimaryExprContext):
        logger.debug(f"visitPrimaryExpr at line {ctx.start.line}")
        # Based on the first token, we can determine the type of primary expression.
        token_type = ctx.start.type
        if token_type == hamiltonianDSLParser.TENSORPROD:
            # TensorProd 调用：TensorProd LPAREN expressionList RPAREN
            expr_list_node = self.visit(ctx.expressionList())
            factors = expr_list_node.expressions
            node = TensorProdNode(
                factor_exprs=factors,
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug("Created TensorProdNode")
            return node
        elif token_type == hamiltonianDSLParser.LPAREN:
            # It's a parenthesized expression: LPAREN expression RPAREN
            expr = self.visit(ctx.expression())
            node = ParenExpressionNode(
                expression=expr,
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug("Created ParenExpressionNode")
            return node
        elif ctx.accumulationExpr():
            return self.visit(ctx.accumulationExpr())
        elif ctx.quantumOp():
            return self.visit(ctx.quantumOp())
        elif ctx.IMAG():
            node = ImagNode(
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug("Created ImagNode")
            return node
        elif ctx.NUMBER():
            text = ctx.NUMBER().getText()
            try:
                if '.' in text or 'e' in text.lower():
                    val = float(text)
                else:
                    val = int(text)
            except ValueError as e:
                logger.error(f"Invalid number format at line {ctx.start.line}, column {ctx.start.column}: {text}")
                raise Exception(f"Invalid number format at line {ctx.start.line}, column {ctx.start.column}: {text}") from e
            node = NumberLiteralNode(
                value=val,
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug(f"Created NumberLiteralNode with value {val}")
            return node
        elif ctx.IDENTIFIER():
            name = ctx.IDENTIFIER().getText()
            node = IdentifierNode(
                name=name,
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug(f"Created IdentifierNode for '{name}'")
            return node
        else:
            logger.error("Unrecognized primary expression")
            raise Exception("Unrecognized primary expression")
    # ----------------------------------------------------------------

    def visitExpressionList(self, ctx: hamiltonianDSLParser.ExpressionListContext):
        logger.debug(f"visitExpressionList at line {ctx.start.line}")
        exprs = [self.visit(e) for e in ctx.expression()]
        node = ExpressionListNode(
            expressions=exprs,
            line=ctx.start.line,
            column=ctx.start.column
        )
        logger.debug("Created ExpressionListNode")
        return node

    def visitAccumulationExpr(self, ctx: hamiltonianDSLParser.AccumulationExprContext):
        logger.debug(f"visitAccumulationExpr at line {ctx.start.line}")
        if ctx.SUM_OVER():
            op_type = AccumulationType.SUM_OVER
        elif ctx.PROD_OVER():
            op_type = AccumulationType.PROD_OVER
        elif ctx.TENSORPROD_OVER():
            op_type = AccumulationType.TENSORPROD_OVER
        else:
            logger.error(f"Unknown accumulation operator at line {ctx.start.line}, column {ctx.start.column}")
            raise Exception(f"Unknown accumulation operator at line {ctx.start.line}, column {ctx.start.column}")
        range_vars = self.visit(ctx.rangeVars())
        body_expr = self.visit(ctx.expression())
        node = AccumulationNode(
            op=op_type,
            range_vars=range_vars,
            body_expr=body_expr,
            line=ctx.start.line,
            column=ctx.start.column
        )
        logger.debug(f"Created AccumulationNode of type {op_type.value}")
        return node

    def visitRangeVars(self, ctx: hamiltonianDSLParser.RangeVarsContext):
        logger.debug(f"visitRangeVars at line {ctx.start.line if ctx.start else 'unknown'}")
        rangeVarNodes = []
        for rv_ctx in ctx.rangeVar():
            rv_node = self.visit(rv_ctx)
            rangeVarNodes.append(rv_node)
            logger.debug(f"Added RangeVarNode for '{rv_node.var_name}'")
        return rangeVarNodes

    def visitRangeVar(self, ctx: hamiltonianDSLParser.RangeVarContext):
        logger.debug(f"visitRangeVar at line {ctx.start.line}")
        if ctx.RANGE():
            var_name = ctx.IDENTIFIER().getText()
            start_expr = self.visit(ctx.expression(0))
            end_expr = self.visit(ctx.expression(1))
            step_expr = self.visit(ctx.expression(2))
            node = RangeVarNode(
                var_name=var_name,
                start_expr=start_expr,
                end_expr=end_expr,
                step_expr=step_expr,
                is_inline=True,
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug(f"Created inline RangeVarNode for '{var_name}'")
            return node
        else:
            var_name = ctx.IDENTIFIER().getText()
            node = RangeVarNode(
                var_name=var_name,
                start_expr=None,
                end_expr=None,
                step_expr=None,
                is_inline=False,
                line=ctx.start.line,
                column=ctx.start.column
            )
            logger.debug(f"Created simple RangeVarNode for '{var_name}'")
            return node

    def visitQuantumOp(self, ctx: hamiltonianDSLParser.QuantumOpContext):
        logger.debug(f"visitQuantumOp at line {ctx.start.line}")
        if ctx.FC():
            op_type = "FC"
        elif ctx.FA():
            op_type = "FA"
        elif ctx.FN():
            op_type = "FN"
        elif ctx.BC():
            op_type = "BC"
        elif ctx.BA():
            op_type = "BA"
        elif ctx.PAULIX():
            op_type = "Pauli_X"
        elif ctx.PAULIY():
            op_type = "Pauli_Y"
        elif ctx.PAULIZ():
            op_type = "Pauli_Z"
        elif ctx.PAULII():
            op_type = "Pauli_I"
        else:
            logger.error("Unknown quantum operation")
            raise Exception("Unknown quantum operation")
        indices = self.visit(ctx.bracketedIndices())
        node = QuantumOpNode(
            op_type=op_type,
            indices=indices,
            line=ctx.start.line,
            column=ctx.start.column
        )
        logger.debug(f"Created QuantumOpNode for '{op_type}'")
        return node

    def visitBracketedIndices(self, ctx: hamiltonianDSLParser.BracketedIndicesContext):
        logger.debug(f"visitBracketedIndices at line {ctx.start.line}")
        expr = self.visit(ctx.expression())
        result = [expr]
        for child in ctx.bracketedIndices():
            tail = self.visit(child)
            if isinstance(tail, list):
                result.extend(tail)
            else:
                result.append(tail)
        logger.debug(f"Bracketed indices: {len(result)} indices found")
        return result

# -----------------------------------------------------------------------------
# Helper function to pretty-print the AST in a tree-like format.
# -----------------------------------------------------------------------------
def pretty_print_ast_tree(node, prefix="", is_last=True):
    connector = "`-- " if is_last else "|-- "
    node_info = node.__class__.__name__
    if hasattr(node, 'name') and node.name is not None:
        node_info += f"({node.name})"
    elif hasattr(node, 'op'):
        node_info += f"({node.op.value})"
    elif hasattr(node, 'value'):
        node_info += f"({node.value})"
    
    pos_info = f" [line: {node.line}, col: {node.column}]" if hasattr(node, 'line') and hasattr(node, 'column') else ""
    logger.debug(prefix + connector + node_info + pos_info)
    new_prefix = prefix + ("    " if is_last else "|   ")
    
    children = []
    for attr in dir(node):
        if attr.startswith("_") or callable(getattr(node, attr)):
            continue
        child = getattr(node, attr)
        if isinstance(child, (ProgramNode, ConstDeclarationNode, RangeDeclarationNode,
                              AssignmentNode, ExpressionStatementNode, BinaryOpNode,
                              ParenExpressionNode, UnaryMinusNode, TensorProdNode,
                              AccumulationNode, RangeVarNode, QuantumOpNode,
                              ImagNode, NumberLiteralNode, IdentifierNode, ExpressionListNode)):
            children.append(child)
        elif isinstance(child, list):
            for item in child:
                if hasattr(item, '__class__') and hasattr(item, 'line'):
                    children.append(item)
    
    for index, child in enumerate(children):
        is_last_child = (index == len(children) - 1)
        pretty_print_ast_tree(child, new_prefix, is_last_child)
