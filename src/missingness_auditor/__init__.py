"""
Missingness Audit & Imputation Planner
=======================================
A reusable skill for autonomous data-science agents that diagnoses missing
data before imputation.

Quick start
-----------
    from missingness_auditor import MissingnessAuditor

    auditor = MissingnessAuditor(df, target_col="target")
    results = auditor.run()
    imputed_df = auditor.apply_imputation(df, results["imputation_plan"])
"""

from .auditor import MissingnessAuditor
from .profiler import MissingnessProfiler
from .mechanism import MechanismAuditor
from .structural import StructuralMissingnessDetector
from .planner import ImputationPlanner
from .imputer import Imputer
from .leakage_check import LeakageSafeImputationChecker
from .visualization import MissingnessVisualizer
from .reporting import ReportWriter
from .mice import MICEImputer, pool_rubin, mice_pool_column_means
from .sensitivity import mnar_sensitivity, plot_tipping_point
from .codegen import emit_reproduction_script

__version__ = "0.3.0"
__all__ = [
    "MissingnessAuditor",
    "MissingnessProfiler",
    "MechanismAuditor",
    "StructuralMissingnessDetector",
    "ImputationPlanner",
    "Imputer",
    "LeakageSafeImputationChecker",
    "MissingnessVisualizer",
    "ReportWriter",
    "MICEImputer",
    "pool_rubin",
    "mice_pool_column_means",
    "mnar_sensitivity",
    "plot_tipping_point",
    "emit_reproduction_script",
]
