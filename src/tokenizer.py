import re

TOKEN_SPEC = [
    ('DEF',      r'def\b'),
    ('RETURN',   r'return\b'),
    ('IF',       r'if\b'),
    ('ELIF',     r'elif\b'),
    ('ELSE',     r'else\b'),
    ('WHILE',    r'while\b'),
    ('FOR',      r'for\b'),
    ('IN',       r'in\b'),
    ('RANGE',    r'range\b'),
    ('PRINT',    r'print\b'),
    ('SYSTEM',   r'system\b'),
    ('IMPORT',   r'import\b'),
    ('FROM',     r'from\b'),
    ('AS',       r'as\b'),
    ('FLOAT',    r'\d+\.\d+'),  # float literals
    ('NUMBER',   r'\d+'),
    ('ID',       r'[A-Za-z_][A-Za-z0-9_]*'),
    ('STRING',   r'"[^"]*"'),
    ('DOT',      r'\.'),
    ('OP',       r'==|!=|<=|>=|[+\-*/<>=]'),
    ('LPAREN',   r'\('),
    ('RPAREN',   r'\)'),
    ('COLON',    r':'),
    ('COMMA',    r','),
    ('NEWLINE',  r'\n'),
    ('SKIP',     r'[ \t]+'),
    ('COMMENT',  r'#.*'),
    ('EOF',      r'$'),
]

TOKEN_REGEX = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPEC)
KEYWORDS = {'def', 'return', 'if', 'elif', 'else', 'while', 'for', 'in', 'range', 'print', 'system', 'import', 'from', 'as'}

def tokenize(code):
    tokens = []
    indents = [0]
    lines = code.splitlines()
    for lineno, line in enumerate(lines):
        # Remove trailing comments
        line = re.sub(r'#.*', '', line)
        if not line.strip():
            continue
        # Count leading spaces (indentation)
        indent = len(line) - len(line.lstrip(' '))
        if indent > indents[-1]:
            indents.append(indent)
            tokens.append(('INDENT', ''))
        while indent < indents[-1]:
            indents.pop()
            tokens.append(('DEDENT', ''))
        pos = 0
        while pos < len(line):
            mo = re.match(TOKEN_REGEX, line[pos:])
            if not mo:
                raise SyntaxError(f"Unknown token at line {lineno+1}: {line[pos:]}")
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'FLOAT':
                tokens.append(('FLOAT', float(value)))
            elif kind == 'NUMBER':
                tokens.append(('NUMBER', int(value)))
            elif kind == 'ID':
                if value in KEYWORDS:
                    tokens.append((value.upper(), value))
                else:
                    tokens.append(('ID', value))
            elif kind == 'STRING':
                tokens.append(('STRING', value[1:-1]))
            elif kind == 'COMMENT' or kind == 'SKIP':
                pass
            elif kind == 'NEWLINE':
                pass
            elif kind == 'EOF':
                break
            else:
                tokens.append((kind, value))
            pos += len(value)
        tokens.append(('NEWLINE', ''))
    while len(indents) > 1:
        indents.pop()
        tokens.append(('DEDENT', ''))
    tokens.append(('EOF', ''))
    return tokens