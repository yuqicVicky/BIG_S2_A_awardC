"""Verify that src/ and .claude/skills/ contain no demo-specific hardcoding.

Forbidden terms may appear in examples/, tests/, or README — never in src/ or the
skill definition files.  This test prevents accidental coupling to any specific
dataset or domain.
"""

import os
import re
import pytest

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC_DIR = os.path.join(_REPO_ROOT, "src", "pattern_auditor")
SKILLS_DIR = os.path.join(_REPO_ROOT, ".claude", "skills")

# Terms forbidden in src/ and .claude/skills/ core logic.
# Examples, tests, and README are explicitly exempted.
FORBIDDEN_TERMS = [
    r"\bbike\b",
    r"\bcasual\b",
    r"\bregistered\b",
    r"\bcnt\b",
    r"\brentals?\b",
    r"\bweathersit\b",
    r"\bSalePrice\b",
    r"\bHousePrice\b",
    r"\bhouse_?price\b",
    r"\bmedian_house\b",
    r"\boverdose\b",
    r"\bfentanyl\b",
    r"\bAmes\b",
    r"\btitanic\b",
    r"\bpassenger\b",
    r"\bsurvived?\b",
    r"\biris\b",
    r"\bspecies\b",
    r"\bsetosa\b",
    r"\bversicolor\b",
    r"\bvirginica\b",
    r"\btarget_component_1\b",
    r"\btarget_component_2\b",
]


def _iter_py_files(directory: str):
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".py") or fname.endswith(".md"):
                yield os.path.join(root, fname)


def test_no_domain_terms_in_src():
    violations = []
    for path in _iter_py_files(SRC_DIR):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for pattern in FORBIDDEN_TERMS:
            for m in re.finditer(pattern, content, re.IGNORECASE):
                line_no = content[: m.start()].count("\n") + 1
                violations.append(
                    f"{os.path.relpath(path)}: line {line_no}: "
                    f"'{m.group()}' (pattern: {pattern})"
                )
    if violations:
        pytest.fail("Dataset-specific hardcoding found in src/:\n" + "\n".join(violations))


def test_no_domain_terms_in_skills():
    if not os.path.isdir(SKILLS_DIR):
        pytest.skip("No .claude/skills/ directory found")
    violations = []
    for path in _iter_py_files(SKILLS_DIR):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for pattern in FORBIDDEN_TERMS:
            for m in re.finditer(pattern, content, re.IGNORECASE):
                line_no = content[: m.start()].count("\n") + 1
                violations.append(
                    f"{os.path.relpath(path)}: line {line_no}: "
                    f"'{m.group()}' (pattern: {pattern})"
                )
    if violations:
        pytest.fail("Domain-specific terms found in .claude/skills/:\n" + "\n".join(violations))


def test_src_files_exist():
    files = list(_iter_py_files(SRC_DIR))
    assert len(files) >= 5, f"Expected ≥5 source files; found {len(files)}"


def test_generic_column_names_used_in_src():
    generic = ["numeric_continuous", "categorical", "datetime_like", "target_col"]
    src_content = ""
    for path in _iter_py_files(SRC_DIR):
        with open(path, encoding="utf-8") as f:
            src_content += f.read()
    for term in generic:
        assert term in src_content, f"Generic term '{term}' not found in src/ — is the code dataset-agnostic?"
