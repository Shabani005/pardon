import re
from llvmlite import ir, binding
import ctypes

# --- Lexer and Parser (very basic, not robust) ---

def tokenize(code):
    # Split by whitespace and symbols
    tokens = re.findall(r'\w+|==|!=|<=|>=|[{}();=+\-*/<>]', code)
    return tokens

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, val):
        tok = self.next()
        if tok != val:
            raise SyntaxError(f"Expected {val}, got {tok}")

    def parse(self):
        functions = {}
        while self.peek():
            if self.peek() == 'fn':
                fn = self.parse_function()
                functions[fn['name']] = fn
            else:
                self.next()  # skip
        return functions

    def parse_function(self):
        self.expect('fn')
        name = self.next()
        self.expect('(')
        params = []
        if self.peek() != ')':
            while True:
                params.append(self.next())
                if self.peek() == ',':
                    self.next()
                else:
                    break
        self.expect(')')
        self.expect('{')
        body = self.parse_block()
        return {'name': name, 'params': params, 'body': body}

    def parse_block(self):
        stmts = []
        while self.peek() and self.peek() != '}':
            stmts.append(self.parse_stmt())
        self.expect('}')
        return stmts

    def parse_stmt(self):
        tok = self.peek()
        if tok == 'let':
            self.next()
            var = self.next()
            self.expect('=')
            expr = self.parse_expr()
            self.expect(';')
            return ('let', var, expr)
        elif tok == 'print':
            self.next()
            self.expect('(')
            expr = self.parse_expr()
            self.expect(')')
            self.expect(';')
            return ('print', expr)
        elif tok == 'for':
            self.next()
            var = self.next()
            self.expect('=')
            start = self.parse_expr()
            self.expect('to')
            end = self.parse_expr()
            self.expect('{')
            body = self.parse_block()
            return ('for', var, start, end, body)
        elif tok == 'while':
            self.next()
            cond = self.parse_expr()
            self.expect('{')
            body = self.parse_block()
            return ('while', cond, body)
        elif tok == 'return':
            self.next()
            expr = self.parse_expr()
            self.expect(';')
            return ('return', expr)
        else:
            # Assignment or expr
            var = self.next()
            if self.peek() == '=':
                self.next()
                expr = self.parse_expr()
                self.expect(';')
                return ('assign', var, expr)
            else:
                raise SyntaxError(f"Unknown statement: {tok}")

    def parse_expr(self):
        # Only support simple binary expressions for now
        left = self.next()
        if self.peek() in ('+', '-', '*', '/', '<', '>', '==', '!=', '<=', '>='):
            op = self.next()
            right = self.next()
            return ('binop', op, left, right)
        else:
            return ('var', left) if left.isidentifier() else ('const', int(left))

# --- Code Generation ---

class CodeGen:
    def __init__(self):
        self.module = ir.Module(name="pardon")
        self.printf = None
        self.funcs = {}
        self.builder = None
        self.locals = {}
        self._fmt_str = None  # <-- Add this line

    def declare_printf(self):
            voidptr_ty = ir.IntType(8).as_pointer()
            printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
            self.printf = ir.Function(self.module, printf_ty, name="printf")
    
    def get_fmt_str(self):
        # Only create the global string once
        if self._fmt_str is None:
            fmt_bytes = bytearray("%d\n".encode("utf8")) + b"\x00"
            fmt_type = ir.ArrayType(ir.IntType(8), len(fmt_bytes))
            self._fmt_str = ir.GlobalVariable(self.module, fmt_type, name="fstr")
            self._fmt_str.linkage = "internal"
            self._fmt_str.global_constant = True
            self._fmt_str.initializer = ir.Constant(fmt_type, fmt_bytes)
        return self._fmt_str

    def codegen(self, functions):
        self.declare_printf()
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
        self.locals = {name: arg for name, arg in zip(fn['params'], func.args)}
        retval = self.codegen_block(fn['body'])
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
        elif kind == 'assign':
            _, var, expr = stmt
            val = self.codegen_expr(expr)
            ptr = self.locals[var]
            self.builder.store(val, ptr)
        elif kind == 'print':
            _, expr = stmt
            val = self.codegen_expr(expr)
            fmt = self.get_fmt_str()
            fmt_ptr = self.builder.bitcast(fmt, ir.IntType(8).as_pointer())
            self.builder.call(self.printf, [fmt_ptr, val])
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
            cond = self.builder.icmp_signed('<=' , cur_val, end_val)
            self.builder.cbranch(cond, loop_body, loop_end)
            self.builder.position_at_start(loop_body)
            old_ptr = self.locals.get(var)
            self.locals[var] = var_ptr
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
            ptr = self.locals[expr[1]]
            return self.builder.load(ptr)
        elif expr[0] == 'binop':
            op, left, right = expr[1], expr[2], expr[3]
            lval = self.codegen_expr(('var', left) if left.isidentifier() else ('const', int(left)))
            rval = self.codegen_expr(('var', right) if right.isidentifier() else ('const', int(right)))
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

# --- Usage Example ---

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pardon_compiler.py <filename.pn>")
        sys.exit(1)
    compile_and_run_pn(sys.argv[1])

