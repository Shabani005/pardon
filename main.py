import re
import sys
from llvmlite import ir, binding
import ctypes

# --- Lexer ---

TOKEN_SPEC = [
    ('FOR',      r'for'),
    ('WHILE',    r'while'),
    ('IF',       r'if'),
    ('ELSE',     r'else'),
    ('RETURN',   r'return'),
    ('LET',      r'let'),
    ('FN',       r'fn'),
    ('PRINT',    r'print'),
    ('TO',       r'to'),
    ('NUMBER',   r'\d+'),
    ('ID',       r'[A-Za-z_][A-Za-z0-9_]*'),
    ('STRING',   r'"[^"]*"'),
    ('COMMENT',  r'//.*'),
    ('SKIP',     r'[ \t]+'),
    ('NEWLINE',  r'\n'),
    ('OP',       r'==|!=|<=|>=|[+\-*/<>=]'),
    ('LPAREN',   r'\('),
    ('RPAREN',   r'\)'),
    ('LBRACE',   r'\{'),
    ('RBRACE',   r'\}'),
    ('COMMA',    r','),
    ('SEMICOL',  r';'),
    ('TO',       r'to'),
    ('IF',       r'if'),
    ('ELSE',     r'else'),
    ('RETURN',   r'return'),
    ('LET',      r'let'),
    ('FN',       r'fn'),
    ('PRINT',    r'print'),
    ('EOF',      r'$'),
]

TOKEN_REGEX = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPEC)
KEYWORDS = {'if', 'else', 'return', 'let', 'fn', 'print', 'to'}

def tokenize(code):
    tokens = []
    for mo in re.finditer(TOKEN_REGEX, code):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'NUMBER':
            tokens.append(('NUMBER', int(value)))
        elif kind == 'ID':
            if value in KEYWORDS:
                tokens.append((value.upper(), value))
            else:
                tokens.append(('ID', value))
        elif kind == 'STRING':
            tokens.append(('STRING', value[1:-1]))
        elif kind == 'COMMENT':
            continue
        elif kind == 'SKIP' or kind == 'NEWLINE':
            continue
        elif kind == 'EOF':
            break
        else:
            tokens.append((kind, value))
    tokens.append(('EOF', ''))
    return tokens

# --- Parser ---

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

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
        functions = {}
        while self.peek()[0] != 'EOF':
            if self.peek()[0] == 'FN':
                fn = self.parse_function()
                functions[fn['name']] = fn
            else:
                self.next()  # skip
        return functions

    def parse_function(self):
        self.expect('FN')
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
        self.expect('LBRACE')
        body = self.parse_block()
        return {'name': name, 'params': params, 'body': body}

    def parse_block(self):
        stmts = []
        while self.peek()[0] != 'RBRACE':
            stmts.append(self.parse_stmt())
        self.expect('RBRACE')
        return stmts

    def parse_stmt(self):
        tok = self.peek()
        if tok[0] == 'LET':
            self.next()
            var = self.expect('ID')[1]
            self.expect('OP')  # '='
            expr = self.parse_expr()
            self.expect('SEMICOL')
            return ('let', var, expr)
        elif tok[0] == 'PRINT':
            self.next()
            self.expect('LPAREN')
            expr = self.parse_expr()
            self.expect('RPAREN')
            self.expect('SEMICOL')
            return ('print', expr)
        elif tok[0] == 'IF':
            self.next()
            cond = self.parse_expr()
            self.expect('LBRACE')
            then_body = self.parse_block()
            else_body = []
            if self.peek()[0] == 'ELSE':
                self.next()
                self.expect('LBRACE')
                else_body = self.parse_block()
            return ('if', cond, then_body, else_body)
        elif tok[0] == 'FOR':
            self.next()
            var = self.expect('ID')[1]
            self.expect('OP')  # '='
            start = self.parse_expr()
            self.expect('TO')
            end = self.parse_expr()
            self.expect('LBRACE')
            body = self.parse_block()
            return ('for', var, start, end, body)
        elif tok[0] == 'WHILE':
            self.next()
            cond = self.parse_expr()
            self.expect('LBRACE')
            body = self.parse_block()
            return ('while', cond, body)
        elif tok[0] == 'RETURN':
            self.next()
            expr = self.parse_expr()
            self.expect('SEMICOL')
            return ('return', expr)
        else:
            # Assignment or expr
            var = self.expect('ID')[1]
            if self.peek()[0] == 'OP' and self.peek()[1] == '=':
                self.next()
                expr = self.parse_expr()
                self.expect('SEMICOL')
                return ('assign', var, expr)
            else:
                raise SyntaxError(f"Unknown statement: {tok}")

    # Pratt parser for expressions with precedence and parentheses
    def parse_expr(self, min_prec=0):
        tok = self.peek()
        if tok[0] == 'NUMBER':
            self.next()
            left = ('const', tok[1])
        elif tok[0] == 'STRING':
            self.next()
            left = ('string', tok[1])
        elif tok[0] == 'ID':
            if self.peek(1)[0] == 'LPAREN':
                # Function call
                name = self.next()[1]
                self.expect('LPAREN')
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
                left = ('var', self.next()[1])
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

# --- Code Generation ---

class CodeGen:
    def __init__(self):
        self.module = ir.Module(name="pardon")
        self.printf = None
        self.scanf = None
        self.funcs = {}
        self.builder = None
        self.locals = {}
        self._fmt_str = {}
        self._input_buf = None

    def declare_printf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    def declare_scanf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        scanf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.scanf = ir.Function(self.module, scanf_ty, name="scanf")

    def get_fmt_str(self, fmt):
        if fmt not in self._fmt_str:
            fmt_bytes = bytearray(fmt.encode("utf8")) + b"\x00"
            fmt_type = ir.ArrayType(ir.IntType(8), len(fmt_bytes))
            gvar = ir.GlobalVariable(self.module, fmt_type, name=f"fstr{len(self._fmt_str)}")
            gvar.linkage = "internal"
            gvar.global_constant = True
            gvar.initializer = ir.Constant(fmt_type, fmt_bytes)
            self._fmt_str[fmt] = gvar
        return self._fmt_str[fmt]

    def codegen(self, functions):
        self.declare_printf()
        self.declare_scanf()
        for name, fn in functions.items():
            self.codegen_function(fn)
        return self.module

    def codegen_function(self, fn):
        fnty = ir.FunctionType(ir.IntType(32), [ir.IntType(32)] * len(fn['params']))
        func = ir.Function(self.module, fnty, name=fn['name'])
        self.funcs[fn['name']] = func
        block = func.append_basic_block('entry')
        builder = ir.IRBuilder(block)
        old_builder = self.builder
        old_locals = self.locals.copy()
        self.builder = builder
        self.locals = {}
        for name, arg in zip(fn['params'], func.args):
            ptr = self.builder.alloca(ir.IntType(32), name=name)
            self.builder.store(arg, ptr)
            self.locals[name] = ptr
        self.var_inited = set(self.locals.keys())
        self.codegen_block(fn['body'])
        if not builder.block.is_terminated:
            builder.ret(ir.Constant(ir.IntType(32), 0))
        self.builder = old_builder
        self.locals = old_locals

    def codegen_block(self, stmts):
        for stmt in stmts:
            self.codegen_stmt(stmt)

    def codegen_stmt(self, stmt):
        kind = stmt[0]
        if kind == 'let':
            _, var, expr = stmt
            val = self.codegen_expr(expr)
            ptr = self.builder.alloca(ir.IntType(32), name=var)
            self.builder.store(val, ptr)
            self.locals[var] = ptr
            self.var_inited.add(var)
        elif kind == 'assign':
            _, var, expr = stmt
            if var not in self.locals:
                raise RuntimeError(f"Variable '{var}' not declared")
            val = self.codegen_expr(expr)
            ptr = self.locals[var]
            self.builder.store(val, ptr)
        elif kind == 'print':
            _, expr = stmt
            val = self.codegen_expr(expr)
            if isinstance(expr, tuple) and expr[0] == 'string':
                fmt = self.get_fmt_str("%s\n")
                fmt_ptr = self.builder.bitcast(fmt, ir.IntType(8).as_pointer())
                str_ptr = self.codegen_expr(expr)
                self.builder.call(self.printf, [fmt_ptr, str_ptr])
            else:
                fmt = self.get_fmt_str("%d\n")
                fmt_ptr = self.builder.bitcast(fmt, ir.IntType(8).as_pointer())
                self.builder.call(self.printf, [fmt_ptr, val])
        elif kind == 'if':
            _, cond, then_body, else_body = stmt
            then_bb = self.builder.append_basic_block('if.then')
            else_bb = self.builder.append_basic_block('if.else')
            end_bb = self.builder.append_basic_block('if.end')
            cond_val = self.codegen_expr(cond)
            self.builder.cbranch(cond_val, then_bb, else_bb)
            # Then
            self.builder.position_at_start(then_bb)
            self.codegen_block(then_body)
            if not self.builder.block.is_terminated:
                self.builder.branch(end_bb)
            # Else
            self.builder.position_at_start(else_bb)
            self.codegen_block(else_body)
            if not self.builder.block.is_terminated:
                self.builder.branch(end_bb)
            # End
            self.builder.position_at_start(end_bb)
        elif kind == 'for':
            _, var, start, end, body = stmt
            start_val = self.codegen_expr(start)
            end_val = self.codegen_expr(end)
            var_ptr = self.builder.alloca(ir.IntType(32), name=var)
            self.builder.store(start_val, var_ptr)
            loop_cond = self.builder.append_basic_block('for.cond')
            loop_body = self.builder.append_basic_block('for.body')
            loop_end = self.builder.append_basic_block('for.end')
            self.builder.branch(loop_cond)
            self.builder.position_at_start(loop_cond)
            cur_val = self.builder.load(var_ptr)
            cond = self.builder.icmp_signed('<=', cur_val, end_val)
            self.builder.cbranch(cond, loop_body, loop_end)
            self.builder.position_at_start(loop_body)
            old_ptr = self.locals.get(var)
            self.locals[var] = var_ptr
            self.var_inited.add(var)
            self.codegen_block(body)
            next_val = self.builder.load(var_ptr)
            next_val = self.builder.add(next_val, ir.Constant(ir.IntType(32), 1))
            self.builder.store(next_val, var_ptr)
            self.builder.branch(loop_cond)
            self.builder.position_at_start(loop_end)
            if old_ptr is not None:
                self.locals[var] = old_ptr
        elif kind == 'while':
            _, cond_expr, body = stmt
            loop_cond = self.builder.append_basic_block('while.cond')
            loop_body = self.builder.append_basic_block('while.body')
            loop_end = self.builder.append_basic_block('while.end')
            self.builder.branch(loop_cond)
            self.builder.position_at_start(loop_cond)
            cond = self.codegen_expr(cond_expr)
            self.builder.cbranch(cond, loop_body, loop_end)
            self.builder.position_at_start(loop_body)
            self.codegen_block(body)
            self.builder.branch(loop_cond)
            self.builder.position_at_start(loop_end)
        elif kind == 'return':
            _, expr = stmt
            val = self.codegen_expr(expr)
            self.builder.ret(val)

    def codegen_expr(self, expr):
        if expr[0] == 'const':
            return ir.Constant(ir.IntType(32), expr[1])
        elif expr[0] == 'var':
            var = expr[1]
            if var not in self.locals:
                raise RuntimeError(f"Variable '{var}' not declared")
            return self.builder.load(self.locals[var])
        elif expr[0] == 'string':
            s = expr[1]
            fmt = self.get_fmt_str(s)
            return self.builder.bitcast(fmt, ir.IntType(8).as_pointer())
        elif expr[0] == 'binop':
            op, left, right = expr[1], expr[2], expr[3]
            lval = self.codegen_expr(left)
            rval = self.codegen_expr(right)
            if op == '+':
                return self.builder.add(lval, rval)
            elif op == '-':
                return self.builder.sub(lval, rval)
            elif op == '*':
                return self.builder.mul(lval, rval)
            elif op == '/':
                return self.builder.sdiv(lval, rval)
            elif op == '<':
                return self.builder.icmp_signed('<', lval, rval)
            elif op == '>':
                return self.builder.icmp_signed('>', lval, rval)
            elif op == '<=':
                return self.builder.icmp_signed('<=', lval, rval)
            elif op == '>=':
                return self.builder.icmp_signed('>=', lval, rval)
            elif op == '==':
                return self.builder.icmp_signed('==', lval, rval)
            elif op == '!=':
                return self.builder.icmp_signed('!=', lval, rval)
            else:
                raise NotImplementedError(f"Unknown op {op}")
        elif expr[0] == 'call':
            name, args = expr[1], expr[2]
            if name == 'read':
                # Read integer from stdin
                fmt = self.get_fmt_str("%d")
                fmt_ptr = self.builder.bitcast(fmt, ir.IntType(8).as_pointer())
                if self._input_buf is None:
                    self._input_buf = self.builder.alloca(ir.IntType(32), name="inputbuf")
                self.builder.call(self.scanf, [fmt_ptr, self._input_buf])
                return self.builder.load(self._input_buf)
            elif name in self.funcs:
                fn = self.funcs[name]
                argvals = [self.codegen_expr(arg) for arg in args]
                return self.builder.call(fn, argvals)
            else:
                raise RuntimeError(f"Unknown function: {name}")
        else:
            raise NotImplementedError(f"Unknown expr: {expr}")

# --- Main Compiler Driver ---

def compile_and_run_pn(filename):
    with open(filename) as f:
        code = f.read()
    tokens = tokenize(code)
    parser = Parser(tokens)
    functions = parser.parse()
    codegen = CodeGen()
    module = codegen.codegen(functions)

    # Print LLVM IR
    print("Generated LLVM IR:")
    print(module)

    # JIT and run main()
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = binding.parse_assembly("")
    engine = binding.create_mcjit_compiler(backing_mod, target_machine)
    mod = binding.parse_assembly(str(module))
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    func_ptr = engine.get_function_address("main")
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int32)(func_ptr)
    cfunc()


def compile_to_ll(filename, output_ll):
    with open(filename) as f:
        code = f.read()
    tokens = tokenize(code)
    parser = Parser(tokens)
    functions = parser.parse()
    codegen = CodeGen()
    module = codegen.codegen(functions)
    with open(output_ll, "w") as f:
        f.write(str(module))
    print(f"LLVM IR written to {output_ll}")


# --- Usage Example ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pardon_compiler.py <filename.pn>")
        sys.exit(1)
    compile_to_ll(sys.argv[1], "exprt.ll")
    compile_and_run_pn(sys.argv[1])
