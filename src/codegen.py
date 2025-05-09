from llvmlite import ir

class CodeGen:
    def __init__(self, functions, modules):
        self.module = ir.Module(name="pardon")
        self.printf = None
        self.scanf = None
        self.system = None
        self.funcs = functions  # global functions
        self.modules = modules  # alias -> {funcname: funcdef}
        self.llvm_funcs = {}    # name/alias -> llvm function
        self.builder = None
        self.locals = {}
        self._fmt_str = {}
        self._input_buf = None
        self._input_buf_float = None
        self.var_types = {}  # varname -> 'int' or 'float'

    def declare_printf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    def declare_scanf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        scanf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.scanf = ir.Function(self.module, scanf_ty, name="scanf")

    def declare_system(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        system_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty])
        self.system = ir.Function(self.module, system_ty, name="system")

    def codegen(self):
        self.declare_printf()
        self.declare_scanf()
        self.declare_system()
        # Declare all functions (including modules)
        for name, fn in self.funcs.items():
            self.declare_function(name, fn)
        for alias, modfns in self.modules.items():
            for name, fn in modfns.items():
                self.declare_function((alias, name), fn)
        return self.module

    def declare_function(self, name, fn):
        if isinstance(name, tuple):
            llvm_name = f"{name[0]}__{name[1]}"
        else:
            llvm_name = name
        fnty = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()] * len(fn['params']))
        func = ir.Function(self.module, fnty, name=llvm_name)
        self.llvm_funcs[name] = func

    def codegen_function(self, name, fn):
        func = self.llvm_funcs[name]
        block = func.append_basic_block('entry')
        builder = ir.IRBuilder(block)
        old_builder = self.builder
        old_locals = self.locals.copy()
        self.builder = builder
        self.locals = {}
        for pname, arg in zip(fn['params'], func.args):
            ptr = self.builder.alloca(ir.DoubleType(), name=pname)
            self.builder.store(arg, ptr)
            self.locals[pname] = ptr
        self.var_inited = set(self.locals.keys())
        self.codegen_block(fn['body'])
        if not builder.block.is_terminated:
            builder.ret(ir.Constant(ir.DoubleType(), 0.0))
        self.builder = old_builder
        self.locals = old_locals

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

    def codegen_block(self, stmts):
        for stmt in stmts:
            self.codegen_stmt(stmt)

    def codegen_stmt(self, stmt):
        kind = stmt[0]
        if kind == 'assign':
            _, var, expr = stmt
            val = self.codegen_expr(expr)
            if var not in self.locals:
                ptr = self.builder.alloca(ir.DoubleType(), name=var)
                self.locals[var] = ptr
            ptr = self.locals[var]
            self.builder.store(val, ptr)
            self.var_inited.add(var)
        elif kind == 'print':
            _, expr = stmt
            val = self.codegen_expr(expr)
            if isinstance(expr, tuple) and expr[0] == 'string':
                fmt = self.get_fmt_str("%s\n")
                fmt_ptr = self.builder.bitcast(fmt, ir.IntType(8).as_pointer())
                str_ptr = self.codegen_expr(expr)
                self.builder.call(self.printf, [fmt_ptr, str_ptr])
            else:
                fmt = self.get_fmt_str("%f\n")
                fmt_ptr = self.builder.bitcast(fmt, ir.IntType(8).as_pointer())
                self.builder.call(self.printf, [fmt_ptr, val])
        elif kind == 'if':
            _, cond, then_body, else_body = stmt
            then_bb = self.builder.append_basic_block('if.then')
            else_bb = self.builder.append_basic_block('if.else')
            end_bb = self.builder.append_basic_block('if.end')
            cond_val = self.codegen_expr(cond)
            cond_val = self.builder.fcmp_ordered('!=', cond_val, ir.Constant(ir.DoubleType(), 0.0))
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
            var_ptr = self.builder.alloca(ir.DoubleType(), name=var)
            self.builder.store(start_val, var_ptr)
            loop_cond = self.builder.append_basic_block('for.cond')
            loop_body = self.builder.append_basic_block('for.body')
            loop_end = self.builder.append_basic_block('for.end')
            self.builder.branch(loop_cond)
            self.builder.position_at_start(loop_cond)
            cur_val = self.builder.load(var_ptr)
            cond = self.builder.fcmp_ordered('<', cur_val, end_val)
            self.builder.cbranch(cond, loop_body, loop_end)
            self.builder.position_at_start(loop_body)
            old_ptr = self.locals.get(var)
            self.locals[var] = var_ptr
            self.var_inited.add(var)
            self.codegen_block(body)
            next_val = self.builder.load(var_ptr)
            next_val = self.builder.fadd(next_val, ir.Constant(ir.DoubleType(), 1.0))
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
            cond = self.builder.fcmp_ordered('!=', cond, ir.Constant(ir.DoubleType(), 0.0))
            self.builder.cbranch(cond, loop_body, loop_end)
            self.builder.position_at_start(loop_body)
            self.codegen_block(body)
            self.builder.branch(loop_cond)
            self.builder.position_at_start(loop_end)
        elif kind == 'return':
            _, expr = stmt
            val = self.codegen_expr(expr)
            self.builder.ret(val)
        elif kind == 'call':
            self.codegen_expr(stmt)

    def codegen_expr(self, expr):
        if expr[0] == 'const':
            return ir.Constant(ir.DoubleType(), float(expr[1]))
        elif expr[0] == 'float':
            return ir.Constant(ir.DoubleType(), expr[1])
        elif expr[0] == 'var':
            var = expr[1]
            if isinstance(var, tuple):
                alias, fname = var
                if alias not in self.modules or fname not in self.modules[alias]:
                    raise RuntimeError(f"Unknown function: {alias}.{fname}")
                llvm_func = self.llvm_funcs[(alias, fname)]
                return llvm_func
            else:
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
                return self.builder.fadd(lval, rval)
            elif op == '-':
                return self.builder.fsub(lval, rval)
            elif op == '*':
                return self.builder.fmul(lval, rval)
            elif op == '/':
                return self.builder.fdiv(lval, rval)
            elif op == '<':
                return self.builder.fcmp_ordered('<', lval, rval)
            elif op == '>':
                return self.builder.fcmp_ordered('>', lval, rval)
            elif op == '<=':
                return self.builder.fcmp_ordered('<=', lval, rval)
            elif op == '>=':
                return self.builder.fcmp_ordered('>=', lval, rval)
            elif op == '==':
                return self.builder.fcmp_ordered('==', lval, rval)
            elif op == '!=':
                return self.builder.fcmp_ordered('!=', lval, rval)
            else:
                raise NotImplementedError(f"Unknown op {op}")
        elif expr[0] == 'call':
            name, args = expr[1], expr[2]
            if name == 'read':
                fmt = self.get_fmt_str("%lf")
                fmt_ptr = self.builder.bitcast(fmt, ir.IntType(8).as_pointer())
                if self._input_buf_float is None:
                    self._input_buf_float = self.builder.alloca(ir.DoubleType(), name="inputbuf_float")
                self.builder.call(self.scanf, [fmt_ptr, self._input_buf_float])
                return self.builder.load(self._input_buf_float)
            elif name == 'system':
                if len(args) != 1 or args[0][0] != 'string':
                    raise RuntimeError("system() only supports a single string literal argument")
                s = args[0][1]
                fmt = self.get_fmt_str(s)
                fmt_ptr = self.builder.bitcast(fmt, ir.IntType(8).as_pointer())
                return self.builder.call(self.system, [fmt_ptr])
            elif name == 'abs':
                # Built-in abs for float
                if len(args) != 1:
                    raise RuntimeError("abs() takes exactly one argument")
                val = self.codegen_expr(args[0])
                zero = ir.Constant(ir.DoubleType(), 0.0)
                is_neg = self.builder.fcmp_ordered('<', val, zero)
                neg_val = self.builder.fsub(zero, val)
                return self.builder.select(is_neg, neg_val, val)
            elif isinstance(name, tuple):
                alias, fname = name
                if alias not in self.modules or fname not in self.modules[alias]:
                    raise RuntimeError(f"Unknown function: {alias}.{fname}")
                fn = self.llvm_funcs[(alias, fname)]
                argvals = [self.codegen_expr(arg) for arg in args]
                return self.builder.call(fn, argvals)
            elif name in self.llvm_funcs:
                fn = self.llvm_funcs[name]
                argvals = [self.codegen_expr(arg) for arg in args]
                return self.builder.call(fn, argvals)
            else:
                raise RuntimeError(f"Unknown function: {name}")
        else:
            raise NotImplementedError(f"Unknown expr: {expr}")