"""PBR: the leaf token PB (per-item base pointer) for the fused recursive task (round 22, 2026-11-01).

Every DRAM access `p.<buf>_base.load[width=W](E)` / `p.<buf>_base.store(E, v)`
inside FFTFusedTask0 whose index E is a sum of integer multiples of index
variables is split as E = C + V (C constant). Accesses in the same block (same
indentation, no reassignment of a variable of V in between) that share
(<buf>, V) get one pointer `var pbrK = p.<buf>_base + (V)` before the first of
them and become `pbrK.load[width=W](C)` / `pbrK.store(C, v)`. SPAD (arena)
accesses are left alone: LLVM re-associates arena bases (leaf study, round 20).
"""
import re

_ACCESS = re.compile(r'\bp\.(\w+_base)\.(load\[width=\d+\]|store)\(')
_ASSIGN = re.compile(r'^\s*(?:var\s+)?(\w+)\s*(?:[-+*/%&|^]|//|<<|>>)?=(?!=)')
_SIMPLE = re.compile(r'^[\w\s+\-*()]+$')


def _args_end(line, start):
    depth = 1
    for i in range(start, len(line)):
        depth += {'(': 1, ')': -1}.get(line[i], 0)
        if depth == 0:
            return i
    raise ValueError(line)


def _first_arg_end(line, start):
    depth = 0
    for i in range(start, len(line)):
        c = line[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            return i
    raise ValueError(line)


def _linear(expr):
    """Polynomial of an index expression (+, -, *, parentheses): {monomial: coefficient}."""
    tokens = re.findall(r'\d+|[A-Za-z_]\w*|[-+*()]', expr)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def add(a, b, sign=1):
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, 0) + sign * v
        return {k: v for k, v in out.items() if v}

    def mul(a, b):
        out = {}
        for ka, va in a.items():
            for kb, vb in b.items():
                k = tuple(sorted(ka + kb))
                out[k] = out.get(k, 0) + va * vb
        return {k: v for k, v in out.items() if v}

    def atom():
        tok = take()
        if tok == '(':
            value = sum_()
            assert take() == ')'
            return value
        if tok == '-':
            return mul({(): -1}, atom())
        if tok.isdigit():
            return {(): int(tok)} if int(tok) else {}
        assert re.fullmatch(r'[A-Za-z_]\w*', tok), tok
        return {(tok,): 1}

    def product():
        value = atom()
        while peek() == '*':
            take()
            value = mul(value, atom())
        return value

    def sum_():
        value = product()
        while peek() in ('+', '-'):
            sign = 1 if take() == '+' else -1
            value = add(value, product(), sign)
        return value

    value = sum_()
    assert pos == len(tokens), expr
    return value


def _split(expr):
    """(constant, variable-part text, symbols) or None."""
    if not _SIMPLE.match(expr):
        return None
    poly = _linear(expr)
    const = poly.pop((), 0)
    if not poly:
        return None
    terms = []
    for mono, coef in sorted(poly.items()):
        factor = '*'.join(mono)
        term = factor if abs(coef) == 1 else f'{abs(coef)}*{factor}'
        terms.append(('- ' if coef < 0 else '+ ') + term)
    text = ' '.join(terms)
    text = text[2:] if text.startswith('+ ') else '-' + text[2:]
    return const, text, {name for mono in poly for name in mono}


def apply(src):
    start = src.index('struct FFTFusedTask0(NDPTask):')
    end = src.index('\ndef main() raises:')
    lines = src[start:end].split('\n')
    # pass 1: collect accesses and their groups
    groups, open_groups, accesses = [], {}, []
    for i, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        if line.strip():
            for key in [k for k, g in open_groups.items() if g['indent'] > indent]:
                del open_groups[key]
        for m in _ACCESS.finditer(line):
            arg = m.end()
            stop = _args_end(line, arg) if m[2].startswith('load') else _first_arg_end(line, arg)
            parts = _split(line[arg:stop])
            if parts is None:
                continue
            const, var, symbols = parts
            key = (indent, m[1], var)
            group = open_groups.get(key)
            if group is None:
                group = dict(indent=indent, buf=m[1], var=var, symbols=symbols, first=i, members=0)
                open_groups[key] = group
                groups.append(group)
            group['members'] += 1
            accesses.append((i, m.start(), arg, stop, const, group))
        target = _ASSIGN.match(line)
        if target:
            for key in [k for k, g in open_groups.items() if target[1] in g['symbols']]:
                del open_groups[key]
    # pass 2: rewrite right to left within each line, then insert the pointers
    count = 0
    for group in groups:
        if group['members'] >= 2:
            group['name'] = f'pbr{count}'
            count += 1
    assert count, 'no shared DRAM base found'
    for i, at, arg, stop, const, group in sorted(accesses, key=lambda a: (a[0], -a[1])):
        if 'name' not in group:
            continue
        line = lines[i]
        method = line[at:arg].split('.', 2)[2]
        lines[i] = line[:at] + f'{group["name"]}.{method}{const}' + line[stop:]
    for group in sorted((g for g in groups if 'name' in g), key=lambda g: -g['first']):
        lines.insert(group['first'], ' ' * group['indent'] + f'var {group["name"]} = p.{group["buf"]} + ({group["var"]})')
    return src[:start] + '\n'.join(lines) + src[end:]
