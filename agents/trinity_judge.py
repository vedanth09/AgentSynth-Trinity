from typing import Dict, Any, Optional
import pandas as pd
import logging
from utils.metrics import (
    calculate_average_wasserstein_distance,
    calculate_jensen_shannon_divergence,
    calculate_correlation_similarity,
    train_test_utility_evaluation,
    check_linkability
)

class TrinityJudge:
    """
    The Trinity Evaluation Engine.
    Validates synthetic data against Fidelity, Utility, and Privacy benchmarks.
    Triggers feedback loops if data fails validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TrinityJudge")
        
        # Thresholds
        self.THRESHOLD_WASSERSTEIN = 0.15 # Relaxed slightly for prototype
        self.THRESHOLD_UTILITY_DROP = 20.0 # Percentage
        self.THRESHOLD_LINKABILITY = 0 # Strict

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the full evaluation suite.
        Returns updated state with 'trinity_scorecard' and 'judge_decision'.
        """
        print("[TrinityJudge] Commencing Evaluation (Fidelity, Utility, Privacy)...")
        
        real_data = state.get("raw_data")
        synthetic_data = state.get("safe_data_asset")
        domain = state.get("domain", "General")
        
        if real_data is None or synthetic_data is None:
            return self._abort_state(state, "Missing Data provided to Judge")

        # 1. Fidelity Benchmarks
        wass_dist = calculate_average_wasserstein_distance(real_data, synthetic_data)
        js_div = calculate_jensen_shannon_divergence(real_data, synthetic_data)
        corr_sim = calculate_correlation_similarity(real_data, synthetic_data)
        
        # 2. Utility Benchmarks (Auto-detect target for classification)
        # Heuristic: Pick a categorical column with few unique values as target
        target_col = None
        for col in real_data.select_dtypes(include=['object', 'int']).columns:
            if real_data[col].nunique() < 10 and real_data[col].nunique() > 1:
                target_col = col
                break
        
        if target_col:
            print(f"[TrinityJudge] Running TSTR Utility Check on target: {target_col}")
            utility_metrics = train_test_utility_evaluation(real_data, synthetic_data, target_col)
        else:
            utility_metrics = {"performance_drop": 0.0, "note": "No suitable target found"}

        # 3. Privacy Benchmarks
        linkability = check_linkability(real_data, synthetic_data)
        
        # 4. Scorecard & Decision
        scorecard = {
            "fidelity": {
                "wasserstein": wass_dist,
                "js_divergence": js_div,
                "correlation_similarity": corr_sim
            },
            "utility": utility_metrics,
            "privacy": linkability
        }
        
        decision, note = self._make_decision(scorecard)
        
        print(f"[TrinityJudge] Decision: {decision}. Note: {note}")
        
        state["trinity_scorecard"] = scorecard
        state["judge_decision"] = decision
        state["judge_feedback"] = note
        
        # Increment iteration count to prevent infinite loops
        state["iteration"] = state.get("iteration", 0) + 1
        
        return state

    def _make_decision(self, scorecard: Dict[str, Any]) -> tuple:
        """Logic to Approve or Reject based on thresholds."""
        
        # Privacy Hard Check
        if not scorecard["privacy"]["is_compliant"]:
            return "Reject", f"CRITICAL: Found {scorecard['privacy']['exact_matches']} exact matches. Increase Noise immediately."

        # Fidelity Checks
        if scorecard["fidelity"]["wasserstein"] > self.THRESHOLD_WASSERSTEIN:
             return "Reject", f"Fidelity too low (Wasserstein {scorecard['fidelity']['wasserstein']:.3f} > {self.THRESHOLD_WASSERSTEIN}). Refine constraints."

        # Utility Checks
        util_drop = scorecard["utility"].get("performance_drop", 0.0)
        if util_drop > self.THRESHOLD_UTILITY_DROP:
             return "Reject", f"Utility dropped by {util_drop:.1f}% (Threshold {self.THRESHOLD_UTILITY_DROP}%). Decrease Noise or optimize structure."

        return "Approve", "Data passed all Trinity benchmarks."

    def _abort_state(self, state, msg):
        state["judge_decision"] = "Error"
        state["judge_feedback"] = msg
        return state
