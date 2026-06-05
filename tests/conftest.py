"""Shared fixtures for all tests."""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def within_period_data():
    """Minimal within-period dataset."""
    from examples.make_toy_within_period_data import make_within_period_data
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    return make_within_period_data(n_entities=10, n_periods=20, seed=0)


@pytest.fixture
def iid_data():
    """Minimal iid dataset."""
    rng = np.random.default_rng(99)
    n = 300
    train = pd.DataFrame({
        "feature_a": rng.normal(0, 1, 240),
        "feature_b": rng.normal(5, 2, 240),
        "feature_c": rng.choice(["X", "Y", "Z"], 240),
        "target": rng.normal(10, 3, 240),
    })
    predict = pd.DataFrame({
        "feature_a": rng.normal(0, 1, 60),
        "feature_b": rng.normal(5, 2, 60),
        "feature_c": rng.choice(["X", "Y", "Z"], 60),
    })
    return train, predict


@pytest.fixture
def group_data():
    """Minimal group-split dataset."""
    from examples.make_toy_group_data import make_group_data
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    return make_group_data(n_groups=20, rows_per_group=10, seed=1)
