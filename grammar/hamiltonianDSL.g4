grammar hamiltonianDSL;

program
    : statementList EOF
    ;

statementList
    : statement*
    ;

statement
    : constDeclaration
    | rangeDeclaration
    | IDENTIFIER ASSIGN expression SEMICOLON
    | expression SEMICOLON
    ;

constDeclaration
    : CONST IDENTIFIER ASSIGN expression SEMICOLON
    ;

rangeDeclaration
    : RANGE IDENTIFIER ASSIGN LBRACKET expression COMMA expression COMMA expression RBRACKET SEMICOLON
    ;

// Expressions are decomposed according to operator precedence
expression
    : addExpr
    ;

// Addition and subtraction expressions (lowest precedence)
addExpr
    : mulExpr ((PLUS | MINUS) mulExpr)*
    ;

// Multiplication and division expressions
mulExpr
    : powerExpr ((MULTIPLY | DIVIDE) powerExpr)*
    ;

// Exponentiation (right-associative)
powerExpr
    : unaryExpr (POWER powerExpr)?
    ;

// Unary operations
unaryExpr
    : MINUS unaryExpr
    | primaryExpr
    ;

// Primary expressions: parentheses, quantum operations, accumulation expressions, etc.
primaryExpr
    : LPAREN expression RPAREN
    | TENSORPROD LPAREN expressionList RPAREN
    | accumulationExpr
    | quantumOp
    | IMAG
    | NUMBER
    | IDENTIFIER
    ;

expressionList
    : expression (COMMA expression)*
    ;

// Accumulation expressions, e.g., Sum_over, Prod_over, TensorProd_over
accumulationExpr
    : SUM_OVER LPAREN rangeVars RPAREN LBRACE expression RBRACE
    | PROD_OVER LPAREN rangeVars RPAREN LBRACE expression RBRACE
    | TENSORPROD_OVER LPAREN rangeVars RPAREN LBRACE expression RBRACE
    ;

// List of range variables
rangeVars
    : rangeVar (COMMA rangeVar)*
    ;

// A range variable can be a single identifier or an inline Range definition
rangeVar
    : IDENTIFIER
    | RANGE IDENTIFIER ASSIGN LBRACKET expression COMMA expression COMMA expression RBRACKET
    ;

// Quantum operations, e.g., FC[i][j], Pauli_X[i][sigma], etc.
quantumOp
    : FC bracketedIndices
    | FA bracketedIndices
    | FN bracketedIndices
    | BC bracketedIndices
    | BA bracketedIndices
    | PAULIX bracketedIndices
    | PAULIY bracketedIndices
    | PAULIZ bracketedIndices
    | PAULII bracketedIndices
    ;

// Bracketed index expressions that can be chained
bracketedIndices
    : LBRACKET expression RBRACKET (bracketedIndices)*
    ;

/*------------------------------------------------------------------
 *  LEXER RULES
 *------------------------------------------------------------------*/

CONST           : 'Const' ;
RANGE           : 'Range' ;
SUM_OVER        : 'Sum_over' ;
PROD_OVER       : 'Prod_over' ;
TENSORPROD      : 'TensorProd' ;
TENSORPROD_OVER : 'TensorProd_over' ;
IMAG            : 'imag' ;

FC              : 'FC' ;
FA              : 'FA' ;
FN              : 'FN' ;
BC              : 'BC' ;
BA              : 'BA' ;

PAULIX          : 'Pauli_X' ;
PAULIY          : 'Pauli_Y' ;
PAULIZ          : 'Pauli_Z' ;
PAULII          : 'Pauli_I' ;

PLUS            : '+' ;
MINUS           : '-' ;
MULTIPLY        : '*' ;
DIVIDE          : '/' ;
POWER           : '^' ;
ASSIGN          : '=' ;

LBRACE          : '{' ;
RBRACE          : '}' ;
LPAREN          : '(' ;
RPAREN          : ')' ;
LBRACKET        : '[' ;
RBRACKET        : ']' ;
SEMICOLON       : ';' ;
COMMA           : ',' ;

NUMBER
    : [0-9]+ ('.' [0-9]+)? ([eE] [+\-]? [0-9]+)?
    ;

IDENTIFIER
    : [a-zA-Z_] [a-zA-Z0-9_]*
    ;

WS
    : [ \t\r\n]+ -> skip
    ;

SINGLECOMMENT
    : '//' ~[\r\n]* -> skip
    ;

MULTICOMMENT
    : '/*' .*? '*/' -> skip
    ;
