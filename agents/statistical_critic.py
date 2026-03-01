from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# SDV Imports
from sdv.single_table import TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Internal Imports
from utils.metrics import calculate_average_wasserstein_distance

# Upgraded Model Imports
from models.timegan_wrapper import ConditionalTimeGANWrapper
from models.ddpm_tabular import DDPMTabular

class ModelLibrary:
    """
    Library of generative models managed by the Statistical Critic.
    """
    
    def __init__(self):
        self.models = {
            "TVAE": TVAESynthesizer,
            "GaussianCopula": GaussianCopulaSynthesizer, 
            "TimeGAN": ConditionalTimeGANWrapper,
            "DDPM": DDPMTabular
        }

    def train_and_evaluate(self, 
                          model_name: str, 
                          real_data: pd.DataFrame, 
                          metadata: SingleTableMetadata) -> tuple:
        """
        Trains and returns (score, synthetic_samples).
        """
        try:
            print(f"   > Piloting {model_name}...")
            SynthesizerClass = self.models.get(model_name)
            
            if not SynthesizerClass:
                return float('inf'), None

            if model_name in ["TimeGAN", "DDPM"]:
                # Custom wrappers
                if model_name == "DDPM":
                    synthesizer = SynthesizerClass(input_dim=real_data.shape[1])
                else:
                    synthesizer = SynthesizerClass() # Internal logic handles dims
                synthesizer.fit(real_data)
                synthetic_data = synthesizer.sample(num_samples=len(real_data))
            else:
                # SDV Synthesizers
                synthesizer = SynthesizerClass(metadata)
                synthesizer.fit(real_data)
                synthetic_data = synthesizer.sample(num_rows=len(real_data))
            
            score = calculate_average_wasserstein_distance(real_data, synthetic_data)
            print(f"     -> {model_name} Score: {score:.4f}")
            
            return score, synthetic_data
        
        except Exception as e:
            print(f"     -> {model_name} Failed: {e}")
            return float('inf'), None

class StatisticalCritic:
    """
    Agent responsible for critiquing logic and selecting optimal generative models
    via an 'Experimental Pilot'. Now supports MODEL ENSEMBLING.
    """
    
    def __init__(self):
        self.library = ModelLibrary()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Academic Motivation: Model ensembling achieves better fidelity-utility 
        trade-offs by combining diverse generative capabilities (Zhao et al., 2018).
        """
        print("[StatisticalCritic] Running Experimental Pilot with Ensembling...")
        
        raw_data: Optional[pd.DataFrame] = state.get("raw_data")
        domain = state.get("domain", "General")
        
        if raw_data is None:
            return state

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(raw_data)

        # 1. Determine Candidates based on Domain
        candidates = ["TVAE", "GaussianCopula"]
        if domain == "Finance": candidates.append("TimeGAN")
        if domain == "Healthcare": candidates.append("DDPM")

        # 2. Run Pilot and collect samples for Ensembling
        pilot_results = {}
        samples_library = {}
        
        data_sample = raw_data.sample(min(len(raw_data), 500))

        for model_name in candidates:
            score, samples = self.library.train_and_evaluate(model_name, data_sample, metadata)
            if samples is not None:
                pilot_results[model_name] = score
                samples_library[model_name] = samples

        # 3. Implement Wasserstein-Weighted Ensembling (Step 2.3)
        # Select Top 2 models and blend them
        sorted_models = sorted(pilot_results.items(), key=lambda x: x[1])
        if len(sorted_models) >= 2:
            m1, s1 = sorted_models[0]
            m2, s2 = sorted_models[1]
            
            # Weighted blending (Inverse of Wasserstein distance: lower score = higher weight)
            w1 = 1.0 / (s1 + 1e-6)
            w2 = 1.0 / (s2 + 1e-6)
            norm_w1 = w1 / (w1 + w2)
            
            print(f"[StatisticalCritic] Ensembling {m1} ({norm_w1:.1%}) and {m2} ({1-norm_w1:.1%})")
            
            # For tabular data, blending can be done via sampling from both distributions
            n1 = int(len(raw_data) * norm_w1)
            n2 = len(raw_data) - n1
            
            ensemble_sample = pd.concat([
                samples_library[m1].sample(n1),
                samples_library[m2].sample(n2)
            ]).reset_index(drop=True)
            
            state["model_selection"] = f"Ensemble({m1}+{m2})"
            state["safe_data_asset"] = ensemble_sample # Primary candidate
        else:
            best_model = sorted_models[0][0]
            state["model_selection"] = best_model
            state["safe_data_asset"] = samples_library[best_model]

        state["pilot_metrics"] = pilot_results
        return state
