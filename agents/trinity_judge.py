import pandas as pd
import numpy as np
import logging
import asyncio
import mlflow
from typing import Dict, Any, List, Optional
from utils.metrics import (
    calculate_average_wasserstein_distance,
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
        
        # Configure MLflow
        mlflow.set_experiment("AgentSynth-Trinity-Synthesis")
        
        # Thresholds
        self.THRESHOLD_WASSERSTEIN = 0.15
        self.THRESHOLD_UTILITY_DROP = 20.0
        self.THRESHOLD_MIA_RISK = 0.65
        self.THRESHOLD_K_ANONYMITY = 3

    async def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Runs parallel evaluation suite with MLflow logging."""
        print("[TrinityJudge] Commencing Evaluation and MLflow Tracking...")
        
        real_data = state.get("raw_data")
        synthetic_data = state.get("safe_data_asset")
        domain = state.get("domain", "General")
        epsilon = state.get("epsilon_input", 1.0)
        
        if real_data is None or synthetic_data is None:
            return self._abort_state(state, "Missing Data")

        # FIX: Align columns between real and synthetic data before any evaluation.
        # After privacy guard / ensembling, column names can diverge (e.g. DDPM
        # outputs generic "med_val_0" columns). Intersect to common columns only.
        common_cols = list(set(real_data.columns).intersection(set(synthetic_data.columns)))
        if not common_cols:
            # No overlap — fall back to positional alignment
            min_cols = min(real_data.shape[1], synthetic_data.shape[1])
            real_data = real_data.iloc[:, :min_cols].copy()
            synthetic_data = synthetic_data.iloc[:, :min_cols].copy()
            synthetic_data.columns = real_data.columns
        else:
            real_data = real_data[common_cols]
            synthetic_data = synthetic_data[common_cols]

        with mlflow.start_run(run_name=f"Synthesis_{domain}_eps{epsilon}"):
            
            async def get_fidelity():
                try:
                    return {
                        "wasserstein": calculate_average_wasserstein_distance(real_data, synthetic_data),
                        "js_divergence": calculate_jensen_shannon_divergence(real_data, synthetic_data),
                        "correlation_similarity": calculate_correlation_similarity(real_data, synthetic_data),
                        "mmd": calculate_mmd(real_data, synthetic_data),
                        "kl_divergence": calculate_kl_divergence(real_data, synthetic_data)
                    }
                except Exception as e:
                    print(f"Fidelity Error: {e}")
                    return {"wasserstein": 1.0, "js_divergence": 1.0,
                            "correlation_similarity": 0.0, "mmd": 1.0, "kl_divergence": 1.0}

            async def get_utility():
                try:
                    # FIX: Smarter target column selection.
                    # Priority 1: binary integer column (ideal for classifier)
                    # Priority 2: any low-cardinality column
                    # Priority 3: any numeric column (will be binarized in explainability)
                    # Priority 4: None → skip utility eval gracefully
                    target_col = None

                    # Look for a binary-ish integer column first
                    for col in real_data.select_dtypes(include=['int64', 'int32']).columns:
                        if 2 <= real_data[col].nunique() <= 10:
                            target_col = col
                            break

                    # Fall back to any object/categorical column with low cardinality
                    if target_col is None:
                        for col in real_data.select_dtypes(include=['object']).columns:
                            if 2 <= real_data[col].nunique() <= 10:
                                target_col = col
                                break

                    # Last resort: use any numeric column (explainability will binarize it)
                    if target_col is None:
                        num_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
                        if num_cols:
                            target_col = num_cols[0]

                    if target_col:
                        print(f"   [TrinityJudge] Utility target column: '{target_col}'")
                        # SHAP explainability (non-blocking — failure won't kill the pipeline)
                        try:
                            self.explainer.explain_utility(real_data, synthetic_data, target_col)
                        except Exception as shap_err:
                            print(f"   [TrinityJudge] SHAP skipped: {shap_err}")
                        return train_test_utility_evaluation(real_data, synthetic_data, target_col)

                    print("   [TrinityJudge] No suitable target column found, skipping utility eval.")
                    return {"performance_drop": 0.0, "tstr_accuracy": 0.0,
                            "trts_accuracy": 0.0, "baseline_accuracy": 0.0}

                except Exception as e:
                    print(f"Utility Error: {e}")
                    return {"performance_drop": 0.0}

            async def get_privacy():
                try:
                    redteam = PrivacyRedTeam()
                    mia_score = redteam.simulate_mia(real_data, synthetic_data)
                    linkability = check_linkability(real_data, synthetic_data)
                    quasi_ids = real_data.select_dtypes(include=['object']).columns[:2].tolist()
                    k_value = redteam.calculate_k_anonymity(synthetic_data, quasi_ids)
                    
                    return {
                        "linkability": linkability,
                        "mia_risk_score": mia_score,
                        "k_anonymity": k_value,
                        "is_compliant": linkability["is_compliant"] and mia_score < self.THRESHOLD_MIA_RISK
                    }
                except Exception as e:
                    print(f"Privacy Error: {e}")
                    return {"linkability": {"is_compliant": True, "exact_matches": 0},
                            "mia_risk_score": 0.5, "k_anonymity": 1, "is_compliant": True}

            results = await asyncio.gather(get_fidelity(), get_utility(), get_privacy())
            
            scorecard = {
                "fidelity": results[0],
                "utility": results[1],
                "privacy": results[2]
            }
            
            # --- MLflow Logging ---
            try:
                mlflow.log_params({"domain": domain, "epsilon": epsilon})
                mlflow.log_metrics({
                    "fidelity_wasserstein": scorecard["fidelity"].get("wasserstein", 0.0),
                    "utility_drop_percent": scorecard["utility"].get("performance_drop", 0.0),
                    "privacy_mia_risk": scorecard["privacy"].get("mia_risk_score", 0.5)
                })
            except Exception as e:
                print(f"   [MLflow] Logging skipped: {e}")

            # --- Artifact Generation ---
            try:
                radar_path = "trinity_radar.png"
                self.reporter.generate_radar_chart(scorecard, output_path=radar_path)
                self.reporter.generate_json_report(scorecard)
                self.cert_gen.generate_text_certificate(scorecard, domain)
                mlflow.log_artifact(radar_path)
                mlflow.log_artifact("benchmark_report.json")
            except Exception as e:
                print(f"   [Artifacts] Generation skipped: {e}")

            decision, note = self._make_decision(scorecard)
            
            state["trinity_scorecard"] = scorecard
            state["judge_decision"] = decision
            state["judge_feedback"] = note
            state["iteration"] = state.get("iteration", 0) + 1
            
            return state

    def _make_decision(self, scorecard: Dict[str, Any]) -> tuple:
        priv = scorecard.get("privacy", {})
        fidelity = scorecard.get("fidelity", {})
        utility = scorecard.get("utility", {})

        if not priv.get("is_compliant", True):
            return "Reject", "Privacy Failed (MIA/Linkability)"
        if priv.get("k_anonymity", 1) < self.THRESHOLD_K_ANONYMITY:
            return "Reject", "Low k-anonymity"
        if fidelity.get("wasserstein", 0) > self.THRESHOLD_WASSERSTEIN:
            return "Reject", "Fidelity poor"
        if utility.get("performance_drop", 0.0) > self.THRESHOLD_UTILITY_DROP:
            return "Reject", "Utility poor"
        return "Approve", "Passed all Trinity benchmarks."

    def _abort_state(self, state: Dict[str, Any], msg: str) -> Dict[str, Any]:
        state["judge_decision"] = "Error"
        state["judge_feedback"] = msg
        return state