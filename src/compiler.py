
import re
import sys
from llvmlite import ir, binding
import ctypes
import subprocess
import os
from tokenizer import tokenize
from pardon_parser import Parser
from codegen import CodeGen


def compile_and_run_pn(filename):
    with open(filename) as f:
        code = f.read()
    tokens = tokenize(code)
    parser = Parser(tokens)
    functions, modules = parser.parse()
    codegen = CodeGen(functions, modules)
    module = codegen.codegen()

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
    cfunc = ctypes.CFUNCTYPE(ctypes.c_double)(func_ptr)
    cfunc()

def compile_to_ll(filename, output_ll):
    with open(filename) as f:
        code = f.read()
    tokens = tokenize(code)
    parser = Parser(tokens)
    functions, modules = parser.parse()
    codegen = CodeGen(functions, modules)
    module = codegen.codegen()
    with open(output_ll, "w") as f:
        f.write(str(module))
    print(f"LLVM IR written to {output_ll}")
    try:
        subprocess.check_call([
            "clang", output_ll, "-o", "runme"
        ])
        print(f"Executable written to runme")
    except subprocess.CalledProcessError as e:
        print(f"Error: clang failed with exit code {e.returncode}")
    os.remove("exprt.ll")

# --- Usage Example ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pardon_compiler.py <filename.pn>")
        sys.exit(1)
    compile_to_ll(sys.argv[1], "exprt.ll")
    compile_and_run_pn(sys.argv[1])
