# src.error_listener

from antlr4.error.ErrorListener import ErrorListener

class HamiltonianErrorListener(ErrorListener):
    def __init__(self):
        super(HamiltonianErrorListener, self).__init__()
        self.errors = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        error = f"Line {line}:{column} - {msg}"
        self.errors.append(error)

    def getErrors(self):
        return self.errors

    def hasErrors(self):
        return len(self.errors) > 0
