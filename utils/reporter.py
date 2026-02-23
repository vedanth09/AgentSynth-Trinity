import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any, List

class TrinityReporter:
    """
    Standardized benchmark reporter for AgentSynth-Trinity.
    Generates JSON reports and Radar Charts for thesis/publication.
    
    Academic Motivation: Radar charts are effective for multi-criteria 
    decision-making in privacy-utility trade-offs, allowing for a holistic 
    view of the 'Trinity' of synthesis performance (Saunders et al., 2019).
    """
    
    @staticmethod
    def generate_json_report(scorecard: Dict[str, Any], output_path: str = "benchmark_report.json"):
        """Saves a structured JSON report."""
        with open(output_path, 'w') as f:
            json.dump(scorecard, f, indent=4)
        print(f"   [Reporter] JSON Benchmark saved to {output_path}")

    @staticmethod
    def generate_radar_chart(scorecard: Dict[str, Any], output_path: str = "trinity_radar.png"):
        """Generates a Radar Chart and returns the figure."""
        labels = ['Fidelity (W)', 'Fidelity (Corr)', 'Utility (TSTR)', 'Utility (TRTS)', 'Privacy (MIA)']
        
        f_w = 1.0 / (1.0 + scorecard['fidelity'].get('wasserstein', 0.5))
        f_c = scorecard['fidelity'].get('correlation_similarity', 0.5)
        u_tstr = scorecard['utility'].get('tstr_accuracy', 0.5)
        u_trts = scorecard['utility'].get('trts_accuracy', 0.5)
        p_mia = 1.0 - (scorecard['privacy'].get('mia_risk_score', 0.5) - 0.5) * 2
        
        values = [f_w, f_c, u_tstr, u_trts, p_mia]
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='#1f77b4', alpha=0.25)
        ax.plot(angles, values, color='#1f77b4', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        
        plt.title("AgentSynth-Trinity: Multi-Dimensional Evaluation", size=15, color='#1f77b4', y=1.1)
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        return fig
