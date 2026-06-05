"""Verify that source code contains no dataset-specific hardcoding.

Checks that domain-specific terms (like those from bike-sharing or similar
well-known datasets) do not appear in the library source code.
Terms are only permitted in examples/, tests/, and README.
"""

import os
import re
import pytest

# Terms that must NOT appear in src/ (domain-specific hardcoding)
FORBIDDEN_IN_SRC = [
    r"\bbike\b",
    r"\bcasual\b",
    r"\bregistered\b",
    r"\bcnt\b",
    r"\brentals?\b",
    r"\bweathersit\b",
    r"\bseason\b",
    r"\bwindspeed\b",
    r"\batemp\b",
    r"\bhumidity\b",
    r"\btitanic\b",
    r"\bpassenger\b",
    r"\bsurvived?\b",
    r"\bhouse_?price\b",
    r"\bmedian_house\b",
    r"\biris\b",
    r"\bspecies\b",
    r"\bsetosa\b",
    r"\bversicolor\b",
    r"\bvirginica\b",
]

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "pattern_auditor")


def _iter_py_files(directory: str):
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".py"):
                yield os.path.join(root, fname)


def test_no_domain_terms_in_src():
    violations = []
    for path in _iter_py_files(SRC_DIR):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for pattern in FORBIDDEN_IN_SRC:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for m in matches:
                # Get line number
                line_no = content[: m.start()].count("\n") + 1
                violations.append(
                    f"{os.path.relpath(path)}: line {line_no}: "
                    f"forbidden term '{m.group()}' (pattern: {pattern})"
                )

    if violations:
        msg = "\n".join(violations)
        pytest.fail(f"Dataset-specific hardcoding found in src/:\n{msg}")


def test_src_files_exist():
    """Sanity check: make sure we found files to scan."""
    files = list(_iter_py_files(SRC_DIR))
    assert len(files) >= 5, f"Expected at least 5 source files, found {len(files)}"


def test_generic_column_names_used():
    """src/ must use generic terms, not domain-specific ones."""
    generic_indicators = [
        "numeric_continuous",
        "categorical",
        "datetime_like",
        "target_col",
    ]
    src_content = ""
    for path in _iter_py_files(SRC_DIR):
        with open(path, encoding="utf-8") as f:
            src_content += f.read()
    for term in generic_indicators:
        assert term in src_content, f"Generic term '{term}' not found in source — is the code dataset-agnostic?"
