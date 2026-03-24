"""
preprocessor.py
---------------
Tokenizes and normalizes Python source code into a list of semantic tokens.
Handles identifier normalization to resist variable-renaming obfuscation.
"""

import tokenize
import io
import re
import keyword
from collections import Counter
from typing import Tuple, List, Dict

# Common Python builtins we want to preserve as meaningful tokens
PYTHON_BUILTINS = {
    'print', 'len', 'range', 'int', 'str', 'float', 'list', 'dict', 'set',
    'tuple', 'bool', 'input', 'open', 'type', 'isinstance', 'hasattr',
    'getattr', 'setattr', 'append', 'extend', 'pop', 'insert', 'remove',
    'return', 'None', 'True', 'False', 'sum', 'max', 'min', 'abs', 'round',
    'sorted', 'reversed', 'enumerate', 'zip', 'map', 'filter', 'any', 'all',
    'iter', 'next', 'super', 'object', 'Exception', 'ValueError', 'TypeError',
    'IndexError', 'KeyError', 'StopIteration', 'NotImplementedError', 'format',
    'join', 'split', 'strip', 'replace', 'find', 'count', 'upper', 'lower',
}

# Operator token map for semantic labeling
OP_MAP = {
    '+': 'OP_ADD',    '-': 'OP_SUB',       '*': 'OP_MUL',   '/': 'OP_DIV',
    '%': 'OP_MOD',    '**': 'OP_POW',      '//': 'OP_FLOORDIV',
    '=': 'OP_ASSIGN', '==': 'OP_EQ',       '!=': 'OP_NEQ',
    '<': 'OP_LT',     '>': 'OP_GT',        '<=': 'OP_LTE',  '>=': 'OP_GTE',
    '(': 'LPAREN',    ')': 'RPAREN',        '[': 'LBRACKET', ']': 'RBRACKET',
    '{': 'LBRACE',    '}': 'RBRACE',        ':': 'COLON',    ',': 'COMMA',
    '.': 'DOT',       '+=': 'OP_IADD',     '-=': 'OP_ISUB', '*=': 'OP_IMUL',
    '/=': 'OP_IDIV',  '->': 'ARROW',       '@': 'DECORATOR', ';': 'SEMICOLON',
    '&': 'OP_AND_BIT','|': 'OP_OR_BIT',    '^': 'OP_XOR',   '~': 'OP_NOT_BIT',
    '<<': 'OP_LSHIFT','>>': 'OP_RSHIFT',
}

SKIP_TYPES = {
    tokenize.COMMENT, tokenize.NEWLINE, tokenize.NL,
    tokenize.ENCODING, tokenize.ENDMARKER,
    tokenize.INDENT, tokenize.DEDENT,
}


def _raw_tokenize(code: str) -> List[Tuple[int, str]]:
    """Internal: extract (tok_type, tok_string) pairs from Python code."""
    result = []
    try:
        readline = io.StringIO(code).readline
        for tok_type, tok_string, _, _, _ in tokenize.generate_tokens(readline):
            if tok_type in SKIP_TYPES or not tok_string.strip():
                continue
            result.append((tok_type, tok_string))
    except tokenize.TokenError:
        # Fallback: simple regex split
        for match in re.finditer(r'\w+|[^\w\s]', code):
            result.append((tokenize.NAME, match.group()))
    return result


def tokenize_python(code: str) -> List[str]:
    """
    Tokenize Python source code into meaningful semantic tokens.

    Keywords are prefixed with KW_, builtins with BUILTIN_,
    operators with OP_, and user identifiers are kept as-is.

    Parameters
    ----------
    code : str  Source code string.

    Returns
    -------
    list[str]  Ordered list of semantic tokens.
    """
    tokens = []
    for tok_type, tok_string in _raw_tokenize(code):
        if tok_type == tokenize.STRING:
            tokens.append('STRING_LITERAL')
        elif tok_type == tokenize.NUMBER:
            tokens.append('NUMBER_LITERAL')
        elif tok_type == tokenize.NAME:
            if keyword.iskeyword(tok_string):
                tokens.append(f'KW_{tok_string}')
            elif tok_string in PYTHON_BUILTINS:
                tokens.append(f'BUILTIN_{tok_string}')
            else:
                tokens.append(tok_string)          # Keep actual name
        elif tok_type == tokenize.OP:
            tokens.append(OP_MAP.get(tok_string, f'OP_MISC'))
    return tokens


def normalize_code(code: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Normalize Python code by replacing user-defined identifiers with
    generic placeholder names (var_0, var_1, …).

    This makes the detector robust against trivial variable renaming.

    Parameters
    ----------
    code : str  Source code string.

    Returns
    -------
    (tokens, identifier_map)
        tokens         : List of normalized tokens.
        identifier_map : Mapping from original name → normalized name.
    """
    tokens = []
    identifier_map: Dict[str, str] = {}
    id_counter = [0]

    for tok_type, tok_string in _raw_tokenize(code):
        if tok_type == tokenize.STRING:
            tokens.append('STR')
        elif tok_type == tokenize.NUMBER:
            tokens.append('NUM')
        elif tok_type == tokenize.NAME:
            if keyword.iskeyword(tok_string) or tok_string in PYTHON_BUILTINS:
                tokens.append(tok_string)
            else:
                if tok_string not in identifier_map:
                    identifier_map[tok_string] = f'var_{id_counter[0]}'
                    id_counter[0] += 1
                tokens.append(identifier_map[tok_string])
        elif tok_type == tokenize.OP:
            tokens.append(OP_MAP.get(tok_string, 'OP_MISC'))

    return tokens, identifier_map


def get_token_stats(tokens: List[str]) -> Dict:
    """
    Compute descriptive statistics about a token list.

    Returns
    -------
    dict with keys: total_tokens, unique_tokens, keywords,
                    builtins, operators, top_10.
    """
    counter = Counter(tokens)
    return {
        'total_tokens':  len(tokens),
        'unique_tokens': len(counter),
        'keywords':  sum(1 for t in tokens if t.startswith('KW_')),
        'builtins':  sum(1 for t in tokens if t.startswith('BUILTIN_')),
        'operators': sum(1 for t in tokens if t.startswith('OP_')),
        'top_10':    counter.most_common(10),
    }
