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
        llm_client=None,
    ):
        self.df = df
        self.predict_df = predict_df
        self.target_col = target_col
        # {col_name: domain_tag} from SKILL.md Step 1.5 semantic analysis.
        # When provided, the planner uses domain-aware strategies (e.g. ffill for
        # weather/sensor/temporal columns instead of median).
        self.domain_tags = domain_tags or {}
        # Optional Anthropic client — when provided, each sub-module enriches its
        # results with LLM-generated narratives, suggestions, and explanations.
        self.llm_client = llm_client
        # Populated after apply_imputation() when llm_client is set.
        self.llm_imputation_summary: str | None = None

    def run(self) -> dict:
        """Run all sub-auditors. Returns a dict of results — no disk writes."""
        profile = MissingnessProfiler(
            self.df, self.predict_df, target_col=self.target_col
        ).profile()

        mechanism_audit = MechanismAuditor(
            self.df, target_col=self.target_col, llm_client=self.llm_client
        ).audit()

        structural_missingness = StructuralMissingnessDetector(
            self.df, llm_client=self.llm_client
        ).detect()

        imputation_plan = ImputationPlanner(
            profile, mechanism_audit, structural_missingness,
            domain_tags=self.domain_tags,
            llm_client=self.llm_client,
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
        """Apply the plan.  Returns imputed DataFrame (or tuple if predict given).

        When llm_client is set, also populates self.llm_imputation_summary with a
        natural language before/after distribution comparison narrative.
        """
        imp = Imputer(llm_client=self.llm_client)
        result = imp.apply(df, imputation_plan, df_predict)
        self.llm_imputation_summary = imp.llm_distribution_summary
        return result

    def save_outputs(self, results: dict, output_dir: str) -> None:
        """Write all JSON logs, figures, reports, and reproducibility artifacts.

        Beyond the core audit, this also writes:
          - logs/mice_pooling.json        Multiple-imputation Rubin pooling (inference)
          - logs/mnar_sensitivity.json    Delta-adjustment MNAR sensitivity
          - figures/mnar_tipping_point.png + figures/decision_flow.png
          - reports/missing_data_report.pdf  Methods-appendix PDF
          - reproduce_imputation.py + source_data.csv  Self-verifying reproduction
        """
        logs_dir = os.path.join(output_dir, "logs")
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

        plan = results["imputation_plan"]

        # Inference-grade artifacts: multiple imputation + MNAR sensitivity.
        # Both are best-effort — a failure here must not block the core report.
        mice_pooling = None
        mnar_sens = None
        try:
            from .mice import mice_pool_column_means
            mice_pooling = mice_pool_column_means(self.df, target_col=self.target_col)
        except Exception as exc:  # pragma: no cover - optional artifact
            print(f"[auditor] MICE pooling skipped: {exc}")
        try:
            from .sensitivity import mnar_sensitivity, plot_tipping_point
            mnar_sens = mnar_sensitivity(self.df, plan, target_col=self.target_col)
            plot_tipping_point(
                mnar_sens, os.path.join(figures_dir, "mnar_tipping_point.png")
            )
        except Exception as exc:  # pragma: no cover - optional artifact
            print(f"[auditor] MNAR sensitivity skipped: {exc}")

        ReportWriter(output_dir, llm_client=self.llm_client).write_all(
            profile=results["missingness_profile"],
            mechanism_audit=results["mechanism_audit"],
            structural_audit=results["structural_missingness"],
            imputation_plan=plan,
            leakage_check=results["leakage_safe_check"],
            mice_pooling=mice_pooling,
            mnar_sensitivity=mnar_sens,
        )

        viz = MissingnessVisualizer(self.df, target_col=self.target_col)
        viz.generate_all(figures_dir=figures_dir)
        try:
            viz.decision_flow(plan, os.path.join(figures_dir, "decision_flow.png"))
        except Exception as exc:  # pragma: no cover - optional artifact
            print(f"[auditor] decision-flow figure skipped: {exc}")

        # Self-verifying reproduction script + a frozen copy of the source data,
        # so the artifact runs standalone months later.
        try:
            from .codegen import emit_reproduction_script
            data_copy = os.path.join(output_dir, "source_data.csv")
            self.df.to_csv(data_copy, index=False)
            emit_reproduction_script(
                plan, data_copy, target_col=self.target_col,
                out_path=os.path.join(output_dir, "reproduce_imputation.py"),
            )
        except Exception as exc:  # pragma: no cover - optional artifact
            print(f"[auditor] reproduction script skipped: {exc}")
