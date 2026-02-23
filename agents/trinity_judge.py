import pandas as pd
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

        with mlflow.start_run(run_name=f"Synthesis_{domain}_eps{epsilon}"):
            # 1. Parallel Metrics
            async def get_fidelity():
                return {
                    "wasserstein": calculate_average_wasserstein_distance(real_data, synthetic_data),
                    "js_divergence": calculate_jensen_shannon_divergence(real_data, synthetic_data),
                    "correlation_similarity": calculate_correlation_similarity(real_data, synthetic_data),
                    "mmd": calculate_mmd(real_data, synthetic_data),
                    "kl_divergence": calculate_kl_divergence(real_data, synthetic_data)
                }

            async def get_utility():
                target_col = None
                for col in real_data.select_dtypes(include=['object', 'int']).columns:
                    if real_data[col].nunique() < 10 and real_data[col].nunique() > 1:
                        target_col = col
                        break
                if target_col:
                    self.explainer.explain_utility(real_data, synthetic_data, target_col)
                    return train_test_utility_evaluation(real_data, synthetic_data, target_col)
                return {"performance_drop": 0.0}

            async def get_privacy():
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

            # Dispatch
            results = await asyncio.gather(get_fidelity(), get_utility(), get_privacy())
            
            scorecard = {
                "fidelity": results[0],
                "utility": results[1],
                "privacy": results[2]
            }
            
            # --- MLflow Logging ---
            mlflow.log_params({"domain": domain, "epsilon": epsilon})
            mlflow.log_metrics({
                "fidelity_wasserstein": scorecard["fidelity"]["wasserstein"],
                "utility_drop_percent": scorecard["utility"].get("performance_drop", 0.0),
                "privacy_mia_risk": scorecard["privacy"]["mia_risk_score"]
            })
            
            # --- Artifact Generation ---
            radar_path = "trinity_radar.png"
            self.reporter.generate_radar_chart(scorecard, output_path=radar_path)
            self.reporter.generate_json_report(scorecard)
            self.cert_gen.generate_text_certificate(scorecard, domain)
            
            mlflow.log_artifact(radar_path)
            mlflow.log_artifact("benchmark_report.json")
            
            decision, note = self._make_decision(scorecard)
            
            state["trinity_scorecard"] = scorecard
            state["judge_decision"] = decision
            state["judge_feedback"] = note
            state["iteration"] = state.get("iteration", 0) + 1
            
            return state

    def _make_decision(self, scorecard: Dict[str, Any]) -> tuple:
        priv = scorecard["privacy"]
        if not priv["is_compliant"]: return "Reject", "Privacy Failed (MIA/Linkability)"
        if priv["k_anonymity"] < self.THRESHOLD_K_ANONYMITY: return "Reject", "Low k-anonymity"
        if scorecard["fidelity"]["wasserstein"] > self.THRESHOLD_WASSERSTEIN: return "Reject", "Fidelity poor"
        if scorecard["utility"].get("performance_drop", 0.0) > self.THRESHOLD_UTILITY_DROP: return "Reject", "Utility poor"
        return "Approve", "Passed all Trinity benchmarks."

    def _abort_state(self, state, msg):
        state["judge_decision"] = "Error"
        state["judge_feedback"] = msg
        return state
