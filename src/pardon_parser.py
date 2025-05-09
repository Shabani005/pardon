from tokenizer import tokenize
import os 

class Parser:
    def __init__(self, tokens, imported_files=None):
        self.tokens = tokens
        self.pos = 0
        self.imported_files = imported_files if imported_files is not None else set()
        self.modules = {}  # alias -> {funcname: funcdef}
        self.functions = {}  # global functions

    def peek(self, k=0):
        if self.pos + k < len(self.tokens):
            return self.tokens[self.pos + k]
        return ('EOF', '')

    def next(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, kind):
        tok = self.next()
        if tok[0] != kind:
            raise SyntaxError(f"Expected {kind}, got {tok}")
        return tok

    def parse(self):
        while self.peek()[0] != 'EOF':
            if self.peek()[0] == 'IMPORT':
                self.next()
                filename = self.expect('STRING')[1]
                alias = None
                if self.peek()[0] == 'AS':
                    self.next()
                    alias = self.expect('ID')[1]
                if self.peek()[0] == 'NEWLINE':
                    self.next()
                if filename in self.imported_files:
                    continue
                self.imported_files.add(filename)
                with open(filename) as f:
                    code = f.read()
                imported_tokens = tokenize(code)
                imported_parser = Parser(imported_tokens, self.imported_files)
                imported_parser.parse()
                if alias is None:
                    alias = os.path.splitext(os.path.basename(filename))[0]
                self.modules[alias] = imported_parser.functions
            elif self.peek()[0] == 'FROM':
                self.next()
                filename = self.expect('STRING')[1]
                self.expect('IMPORT')
                funcname = self.expect('ID')[1]
                asname = funcname
                if self.peek()[0] == 'AS':
                    self.next()
                    asname = self.expect('ID')[1]
                if self.peek()[0] == 'NEWLINE':
                    self.next()
                with open(filename) as f:
                    code = f.read()
                imported_tokens = tokenize(code)
                imported_parser = Parser(imported_tokens, self.imported_files)
                imported_parser.parse()
                if funcname not in imported_parser.functions:
                    raise SyntaxError(f"Function {funcname} not found in {filename}")
                self.functions[asname] = imported_parser.functions[funcname]
            elif self.peek()[0] == 'DEF':
                fn = self.parse_function()
                if fn['name'] in self.functions:
                    raise SyntaxError(f"Function {fn['name']} already defined")
                self.functions[fn['name']] = fn
            else:
                self.next()  # skip
        return self.functions, self.modules

    def parse_function(self):
        self.expect('DEF')
        name = self.expect('ID')[1]
        self.expect('LPAREN')
        params = []
        if self.peek()[0] != 'RPAREN':
            while True:
                params.append(self.expect('ID')[1])
                if self.peek()[0] == 'COMMA':
                    self.next()
                else:
                    break
        self.expect('RPAREN')
        self.expect('COLON')
        self.expect('NEWLINE')
        self.expect('INDENT')
        body = self.parse_block()
        self.expect('DEDENT')
        return {'name': name, 'params': params, 'body': body}

    def parse_block(self):
        stmts = []
        while self.peek()[0] not in ('DEDENT', 'EOF'):
            if self.peek()[0] == 'NEWLINE':
                self.next()
                continue
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self):
        tok = self.peek()
        # Assignment
        if tok[0] == 'ID' and self.peek(1)[0] == 'OP' and self.peek(1)[1] == '=':
            var = self.next()[1]
            self.next()  # '='
            expr = self.parse_expr()
            if self.peek()[0] == 'NEWLINE':
                self.next()
            return ('assign', var, expr)
        # print(...)
        elif tok[0] == 'PRINT':
            self.next()
            self.expect('LPAREN')
            expr = self.parse_expr()
            self.expect('RPAREN')
            if self.peek()[0] == 'NEWLINE':
                self.next()
            return ('print', expr)
        # return ...
        elif tok[0] == 'RETURN':
            self.next()
            expr = self.parse_expr()
            if self.peek()[0] == 'NEWLINE':
                self.next()
            return ('return', expr)
        # if ...
        elif tok[0] == 'IF':
            self.next()
            cond = self.parse_expr()
            self.expect('COLON')
            self.expect('NEWLINE')
            self.expect('INDENT')
            then_body = self.parse_block()
            self.expect('DEDENT')
            else_body = []
            if self.peek()[0] == 'ELSE':
                self.next()
                self.expect('COLON')
                self.expect('NEWLINE')
                self.expect('INDENT')
                else_body = self.parse_block()
                self.expect('DEDENT')
            return ('if', cond, then_body, else_body)
        # while ...
        elif tok[0] == 'WHILE':
            self.next()
            cond = self.parse_expr()
            self.expect('COLON')
            self.expect('NEWLINE')
            self.expect('INDENT')
            body = self.parse_block()
            self.expect('DEDENT')
            return ('while', cond, body)
        # for ...
        elif tok[0] == 'FOR':
            self.next()
            var = self.expect('ID')[1]
            self.expect('IN')
            self.expect('RANGE')
            self.expect('LPAREN')
            start = self.parse_expr()
            self.expect('COMMA')
            end = self.parse_expr()
            self.expect('RPAREN')
            self.expect('COLON')
            self.expect('NEWLINE')
            self.expect('INDENT')
            body = self.parse_block()
            self.expect('DEDENT')
            return ('for', var, start, end, body)
        # function call as statement (including system, dotted)
        elif (tok[0] in ('ID', 'SYSTEM')) or (tok[0] == 'ID' and self.peek(1)[0] == 'DOT'):
            expr = self.parse_expr()
            if self.peek()[0] == 'NEWLINE':
                self.next()
            return expr
        else:
            raise SyntaxError(f"Unknown statement: {tok}")

    def parse_expr(self, min_prec=0):
        tok = self.peek()
        # --- Unary minus/plus support ---
        if tok[0] == 'OP' and tok[1] in ('-', '+'):
            op = self.next()[1]
            expr = self.parse_expr(100)
            if op == '-':
                return ('binop', '-', ('const', 0), expr)
            else:
                return expr
        # --- End unary ---
        if tok[0] == 'FLOAT':
            self.next()
            left = ('float', tok[1])
        elif tok[0] == 'NUMBER':
            self.next()
            left = ('const', tok[1])
        elif tok[0] == 'STRING':
            self.next()
            left = ('string', tok[1])
        elif tok[0] == 'ID' or tok[0] == 'SYSTEM':
            # Dotted name support
            name = self.next()[1]
            while self.peek()[0] == 'DOT':
                self.next()
                name2 = self.expect('ID')[1]
                name = (name, name2)
            if self.peek()[0] == 'LPAREN':
                self.next()
                args = []
                if self.peek()[0] != 'RPAREN':
                    while True:
                        args.append(self.parse_expr())
                        if self.peek()[0] == 'COMMA':
                            self.next()
                        else:
                            break
                self.expect('RPAREN')
                left = ('call', name, args)
            else:
                left = ('var', name)
        elif tok[0] == 'LPAREN':
            self.next()
            left = self.parse_expr()
            self.expect('RPAREN')
        else:
            raise SyntaxError(f"Unexpected token in expression: {tok}")

        # Operator precedence
        while True:
            tok = self.peek()
            if tok[0] == 'OP' and tok[1] in ('+', '-', '*', '/', '<', '>', '<=', '>=', '==', '!='):
                prec = self.get_prec(tok[1])
                if prec < min_prec:
                    break
                op = self.next()[1]
                right = self.parse_expr(prec + 1)
                left = ('binop', op, left, right)
            else:
                break
        return left

    def get_prec(self, op):
        return {
            '==': 1, '!=': 1,
            '<': 2, '>': 2, '<=': 2, '>=': 2,
            '+': 3, '-': 3,
            '*': 4, '/': 4,
        }.get(op, 0)