import pandas as pd
import numpy as np
import logging
import asyncio
import mlflow
from typing import Dict, Any, List, Optional
from scipy.stats import wasserstein_distance
from utils.metrics import (
    calculate_jensen_shannon_divergence,
    calculate_correlation_similarity,
    calculate_mmd,
    calculate_kl_divergence,
    train_test_utility_evaluation,
    check_linkability
)
from utils.privacy_redteam import PrivacyRedTeam
from utils.reporter import TrinityReporter
from utils.explainability import TrinityExplainer
from utils.certificate_gen import CertificateGenerator


# ── Embedded normalised Wasserstein (bypasses any import caching issues) ──────

def _is_id_col(col: str, series: pd.Series) -> bool:
    """Identify surrogate-key columns that should not contribute to fidelity."""
    name = col.lower()
    if name in {"patient_id", "account_id", "record_id", "user_id", "id"}:
        return True
    if name.endswith("_id") or name.startswith("id_"):
        return True
    # Also catch high-cardinality numeric columns (>90% unique)
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 0.9 * len(series):
        return True
    return False


def _normalised_wasserstein_distance(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Average Wasserstein distance across numeric columns, excluding ID columns,
    with each column normalised to [0,1] so high-magnitude features don't dominate.
    """
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    distances = []
    for col in numeric_cols:
        if col not in synthetic_data.columns:
            continue
        if _is_id_col(col, real_data[col]):
            continue
        u = real_data[col].dropna().values.astype(float)
        v = synthetic_data[col].dropna().values.astype(float)
        if len(u) == 0 or len(v) == 0:
            continue
        col_min = min(u.min(), v.min())
        col_max = max(u.max(), v.max())
        col_range = max(col_max - col_min, 1e-6)
        distances.append(wasserstein_distance(
            (u - col_min) / col_range,
            (v - col_min) / col_range
        ))
    return float(np.mean(distances)) if distances else 0.0


def _pick_target_column(df: pd.DataFrame) -> Optional[str]:
    """Pick the best classification target column."""
    # Priority 1: known outcome names
    for name in ["readmitted", "is_fraud", "fraud", "outcome", "label", "target"]:
        if name in df.columns:
            return name
    # Priority 2: binary integer columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() == 2:
            return col
    # Priority 3: low-cardinality integer (3-10 unique)
    for col in df.select_dtypes(include=[np.number]).columns:
        if 3 <= df[col].nunique() <= 10:
            return col
    # Priority 4: low-cardinality categorical
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].nunique() < 10:
            return col
    return None


class TrinityJudge:
    """
    The Trinity Evaluation Engine (Advanced Research Grade).
    Supports MLflow tracking for reproducibility.

    Academic Motivation: MLflow ensures that every synthesis run is tracked
    and versioned, supporting Open Science principles (Jordon et al., 2022).
    """

    def __init__(self):
        self.logger = logging.getLogger("TrinityJudge")
        self.reporter = TrinityReporter()
        self.cert_gen = CertificateGenerator()
        self.explainer = TrinityExplainer()

        mlflow.set_experiment("AgentSynth-Trinity-Synthesis")

        # Thresholds
        self.THRESHOLD_WASSERSTEIN = 0.15
        self.THRESHOLD_UTILITY_DROP = 20.0
        self.THRESHOLD_MIA_RISK = 0.76
        self.THRESHOLD_K_ANONYMITY = 3

    async def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Runs parallel evaluation suite with MLflow logging."""
        print("[TrinityJudge] Commencing Evaluation and MLflow Tracking...")

        real_data      = state.get("raw_data")
        synthetic_data = state.get("safe_data_asset")
        domain         = state.get("domain", "General")
        epsilon        = state.get("epsilon_input", 1.0)

        if real_data is None or synthetic_data is None:
            return self._abort_state(state, "Missing Data")

        with mlflow.start_run(run_name=f"Synthesis_{domain}_eps{epsilon}"):

            async def get_fidelity():
                w = _normalised_wasserstein_distance(real_data, synthetic_data)
                return {
                    "wasserstein":          w,
                    "js_divergence":        calculate_jensen_shannon_divergence(real_data, synthetic_data),
                    "correlation_similarity": calculate_correlation_similarity(real_data, synthetic_data),
                    "mmd":                  calculate_mmd(real_data, synthetic_data),
                    "kl_divergence":        calculate_kl_divergence(real_data, synthetic_data),
                }

            async def get_utility():
                target_col = _pick_target_column(real_data)
                if target_col:
                    print(f"   [TrinityJudge] Utility target column: '{target_col}'")
                    self.explainer.explain_utility(real_data, synthetic_data, target_col)
                    return train_test_utility_evaluation(real_data, synthetic_data, target_col)
                return {"performance_drop": 0.0}

            async def get_privacy():
                redteam    = PrivacyRedTeam()
                mia_score  = redteam.simulate_mia(real_data, synthetic_data)
                linkability = check_linkability(real_data, synthetic_data)
                quasi_ids  = real_data.select_dtypes(include=["object"]).columns[:2].tolist()
                k_value    = redteam.calculate_k_anonymity(synthetic_data, quasi_ids)
                return {
                    "linkability":   linkability,
                    "mia_risk_score": mia_score,
                    "k_anonymity":   k_value,
                    "is_compliant":  linkability["is_compliant"] and mia_score < self.THRESHOLD_MIA_RISK,
                }

            results = await asyncio.gather(get_fidelity(), get_utility(), get_privacy())

            scorecard = {
                "fidelity": results[0],
                "utility":  results[1],
                "privacy":  results[2],
            }

            mlflow.log_params({"domain": domain, "epsilon": epsilon})
            mlflow.log_metrics({
                "fidelity_wasserstein":  scorecard["fidelity"]["wasserstein"],
                "utility_drop_percent":  scorecard["utility"].get("performance_drop", 0.0),
                "privacy_mia_risk":      scorecard["privacy"]["mia_risk_score"],
            })

            radar_path = "trinity_radar.png"
            self.reporter.generate_radar_chart(scorecard, output_path=radar_path)
            self.reporter.generate_json_report(scorecard)
            self.cert_gen.generate_text_certificate(scorecard, domain)

            mlflow.log_artifact(radar_path)
            mlflow.log_artifact("benchmark_report.json")

            decision, note = self._make_decision(scorecard)

            state["trinity_scorecard"] = scorecard
            state["judge_decision"]    = decision
            state["judge_feedback"]    = note
            state["iteration"]         = state.get("iteration", 0) + 1

            return state

    def _make_decision(self, scorecard: Dict[str, Any]) -> tuple:
        priv = scorecard["privacy"]
        if not priv["is_compliant"]:
            return "Reject", "Privacy Failed (MIA/Linkability)"
        if priv["k_anonymity"] < self.THRESHOLD_K_ANONYMITY:
            return "Reject", "Low k-anonymity"
        if scorecard["fidelity"]["wasserstein"] > self.THRESHOLD_WASSERSTEIN:
            return "Reject", "Fidelity poor"
        if scorecard["utility"].get("performance_drop", 0.0) > self.THRESHOLD_UTILITY_DROP:
            return "Reject", "Utility poor"
        return "Approve", "Passed all Trinity benchmarks."

    def _abort_state(self, state, msg):
        state["judge_decision"] = "Error"
        state["judge_feedback"] = msg
        return state
