def tokenize(query):
    tokens = []
    token = ""
    for c in query:
        if c in "()":
            if token:
                tokens.append(token)
                token = ""
            tokens.append(c)
        elif c.isspace():
            if token:
                tokens.append(token)
                token = ""
        else:
            token += c
    if token:
        tokens.append(token)
    return tokens


def parse(tokens):
    def parse_expression(index, min_precedence=0):
        def get_precedence(op):
            return {"OR": 1, "AND": 2, "NOT": 3}.get(op, -1)

        def parse_primary(index):
            token = tokens[index]
            if token == "(":
                subexpr, index = parse_expression(index + 1)
                if tokens[index] != ")":
                    raise ValueError("Expected ')'")
                return subexpr, index + 1
            elif token == "NOT":
                operand, next_index = parse_primary(index + 1)
                return ["NOT", operand], next_index
            else:
                return token, index + 1

        lhs, index = parse_primary(index)

        while index < len(tokens):
            op = tokens[index]
            prec = get_precedence(op)
            if prec < min_precedence:
                break

            index += 1
            rhs, index = parse_expression(index, prec + 1)
            lhs = [op, lhs, rhs]

        return lhs, index

    parsed_tree, next_index = parse_expression(0)
    if next_index != len(tokens):
        raise ValueError("Unexpected tokens remaining")
    return parsed_tree


def convert_query(query):
    tokens = tokenize(query)
    tree = parse(tokens)
    return tree
