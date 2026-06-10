"""Main MissingnessAuditor orchestrator."""

from __future__ import annotations

import os

import pandas as pd

from .profiler import MissingnessProfiler
from .mechanism import MechanismAuditor
from .structural import StructuralMissingnessDetector
from .planner import ImputationPlanner
from .leakage_check import LeakageSafeImputationChecker
from .imputer import Imputer
from .visualization import MissingnessVisualizer
from .reporting import ReportWriter


class MissingnessAuditor:
    """
    Diagnoses missing data before imputation and can apply the recommended plan.

    Quick start
    -----------
    auditor = MissingnessAuditor(df, target_col="target")
    results = auditor.run()
    imputed_df = auditor.apply_imputation(df, results["imputation_plan"])

    Results keys
    ------------
    missingness_profile   : per-column rates, severity, dtype
    mechanism_audit       : MCAR / MAR-like / MNAR clue per column
    structural_missingness: structural absence pairs
    imputation_plan       : per-column strategy, indicator flag, fit scope
    leakage_safe_check    : leakage risk findings + global protocol
    """

    def __init__(
        self,
        df: pd.DataFrame,
        predict_df: pd.DataFrame | None = None,
        *,
        target_col: str | None = None,
        domain_tags: dict[str, str] | None = None,
    ):
        self.df = df
        self.predict_df = predict_df
        self.target_col = target_col
        # {col_name: domain_tag} from SKILL.md Step 1.5 semantic analysis.
        # When provided, the planner uses domain-aware strategies (e.g. ffill for
        # weather/sensor/temporal columns instead of median).
        self.domain_tags = domain_tags or {}

    def run(self) -> dict:
        """Run all sub-auditors. Returns a dict of results — no disk writes."""
        profile = MissingnessProfiler(
            self.df, self.predict_df, target_col=self.target_col
        ).profile()

        mechanism_audit = MechanismAuditor(
            self.df, target_col=self.target_col
        ).audit()

        structural_missingness = StructuralMissingnessDetector(self.df).detect()

        imputation_plan = ImputationPlanner(
            profile, mechanism_audit, structural_missingness,
            domain_tags=self.domain_tags,
        ).plan()

        leakage_safe_check = LeakageSafeImputationChecker(
            imputation_plan, has_predict_df=self.predict_df is not None
        ).check()

        return {
            "missingness_profile": profile,
            "mechanism_audit": mechanism_audit,
            "structural_missingness": structural_missingness,
            "imputation_plan": imputation_plan,
            "leakage_safe_check": leakage_safe_check,
        }

    def apply_imputation(
        self,
        df: pd.DataFrame,
        imputation_plan: dict,
        df_predict: pd.DataFrame | None = None,
    ):
        """Apply the plan.  Returns imputed DataFrame (or tuple if predict given)."""
        return Imputer().apply(df, imputation_plan, df_predict)

    def save_outputs(self, results: dict, output_dir: str) -> None:
        """Write all JSON logs, figures, and markdown report to disk."""
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

        ReportWriter(output_dir).write_all(
            profile=results["missingness_profile"],
            mechanism_audit=results["mechanism_audit"],
            structural_audit=results["structural_missingness"],
            imputation_plan=results["imputation_plan"],
            leakage_check=results["leakage_safe_check"],
        )

        MissingnessVisualizer(self.df, target_col=self.target_col).generate_all(
            figures_dir=os.path.join(output_dir, "figures")
        )
