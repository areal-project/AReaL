# mypy: ignore-errors


class SemanticParserUtil:
    @staticmethod
    def lisp_to_nested_expression(lisp_string: str) -> list:
        """
        Takes a logical form as a lisp string and returns a nested list representation of the lisp.
        For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
        """
        stack: list = []
        current_expression: list = []
        tokens = lisp_string.split()
        for token in tokens:
            while token[0] == "(":
                nested_expression: list = []
                current_expression.append(nested_expression)
                stack.append(current_expression)
                current_expression = nested_expression
                token = token[1:]
            current_expression.append(token.replace(")", ""))
            while token[-1] == ")":
                current_expression = stack.pop()
                token = token[:-1]
        return current_expression[0]

    @staticmethod
    def expression_to_lisp(expression) -> str:
        rtn = "("
        for i, e in enumerate(expression):
            if isinstance(e, list):
                rtn += SemanticParserUtil.expression_to_lisp(e)
            else:
                rtn += e
            if i != len(expression) - 1:
                rtn += " "
        rtn += ")"
        return rtn


def main():
    lisp = "(A ((B C) D E) F)"
    expression = SemanticParserUtil.lisp_to_nested_expression(lisp)
    print(expression)
    print(SemanticParserUtil.expression_to_lisp(expression))


if __name__ == "__main__":
    main()
