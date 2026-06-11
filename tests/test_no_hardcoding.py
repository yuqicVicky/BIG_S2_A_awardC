"""The core (`src/`) must stay dataset-agnostic.

No competition- or demo-specific column name may appear as a string literal used in
*code logic*. Example column names inside docstrings are allowed (they document
behaviour); this test parses the AST and ignores module/class/function docstrings.
"""

import ast
import glob
import os

# Names that would betray hardcoding to a particular dataset.
FORBIDDEN = {
    "overdose", "naloxone", "ed_visit", "ed_visits", "ed_visit_rate",
    "facility_quality", "risk_score", "income", "education_level",
    "pool_type", "pool_area",
}

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "missingness_auditor",
)


def _docstring_node_ids(tree):
    """Collect id()s of string-constant nodes that are docstrings, to skip them."""
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                skip.add(id(body[0].value))
    return skip


def _code_string_literals(path):
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    skip = _docstring_node_ids(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                out.append(node.value.lower())
    return out


def test_no_domain_column_names_in_src_logic():
    py_files = glob.glob(os.path.join(SRC_DIR, "*.py"))
    assert py_files, "no source files found"
    offenders = []
    for path in py_files:
        for literal in _code_string_literals(path):
            for bad in FORBIDDEN:
                # Match the token as a whole word inside the string literal.
                if bad == literal or f"'{bad}'" == literal:
                    offenders.append((os.path.basename(path), bad, literal))
    assert not offenders, f"hardcoded domain column names in src logic: {offenders}"
