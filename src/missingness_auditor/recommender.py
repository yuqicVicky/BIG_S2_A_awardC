"""Column-specific imputation strategy recommendation.

This module exists for backward compatibility. It delegates to ImputationPlanner,
which contains the authoritative strategy-selection logic.
"""

from __future__ import annotations

from .planner import ImputationPlanner


class ImputationRecommender:
    """
    Recommends a per-column imputation strategy based on:
    - missingness profile (rate, dtype, cardinality)
    - mechanism clue (group-dependent / MAR-like / target-associated / MCAR)
    - structural patterns (categorical NA → numeric zero/absent)

    Delegates to ImputationPlanner for all logic.
    """

    def __init__(
        self,
        profile: dict,
        mechanism_audit: dict,
        structural_audit: dict,
    ):
        self._planner = ImputationPlanner(profile, mechanism_audit, structural_audit)

    def recommend(self) -> dict:
        return self._planner.plan()
