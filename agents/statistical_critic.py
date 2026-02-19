from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# SDV Imports
from sdv.single_table import TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Internal Imports
from utils.metrics import calculate_average_wasserstein_distance

class ModelLibrary:
    """
    Library of generative models managed by the Statistical Critic.
    """
    
    def __init__(self):
        self.models = {
            "TVAE": TVAESynthesizer,
            "GaussianCopula": GaussianCopulaSynthesizer, 
            # Note: "Diffusion" logic would go here, using a custom wrapper if needed.
            # Using GaussianCopula as a baseline alternative for now.
        }

    def train_and_evaluate(self, 
                          model_name: str, 
                          real_data: pd.DataFrame, 
                          metadata: SingleTableMetadata) -> float:
        """
        Trains a model on real data and evaluates it against a synthetic sample.
        Returns the Wasserstein distance (lower is better).
        """
        try:
            print(f"   > Piloting {model_name}...")
            SynthesizerClass = self.models.get(model_name)
            
            if not SynthesizerClass:
                return float('inf')

            synthesizer = SynthesizerClass(metadata)
            
            # Quick training for pilot (subset of epochs/steps if configurable)
            # SDV automatic settings usually handle this, but we keep it simple for the pilot.
            synthesizer.fit(real_data)
            
            # Generate sample
            synthetic_data = synthesizer.sample(num_rows=len(real_data))
            
            # Calculate fidelity
            score = calculate_average_wasserstein_distance(real_data, synthetic_data)
            print(f"     -> {model_name} Score: {score:.4f}")
            
            return score
        
        except Exception as e:
            print(f"     -> {model_name} Failed: {e}")
            return float('inf')

class StatisticalCritic:
    """
    Agent responsible for critiquing logic and selecting optimal generative models
    via an 'Experimental Pilot'.
    """
    
    def __init__(self):
        self.library = ModelLibrary()

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the Experimental Pilot to select the best model.
        
        Args:
            state (Dict[str, Any]): The current state.
            
        Returns:
            Dict[str, Any]: Updated state with 'selected_model' and 'pilot_metrics'.
        """
        print("[StatisticalCritic] Running Experimental Pilot...")
        
        raw_data: Optional[pd.DataFrame] = state.get("raw_data")
        
        if raw_data is None:
            print("Error: No raw data found in state.")
            return state

        # Detect Metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(raw_data)

        # Run Pilot
        results = {}
        best_score = float('inf')
        best_model = None

        # Compare TVAE vs GaussianCopula (TimeGAN logic would be similar but requires sequential data structure)
        data_sample = raw_data.sample(min(len(raw_data), 100)) # Use small sample for pilot speed

        for model_name in ["TVAE", "GaussianCopula"]:
            score = self.library.train_and_evaluate(model_name, data_sample, metadata)
            results[model_name] = score
            
            if score < best_score:
                best_score = score
                best_model = model_name

        print(f"[StatisticalCritic] Selected Best Model: {best_model} (Wasserstein: {best_score:.4f})")

        # Update State
        state["model_selection"] = best_model
        state["pilot_metrics"] = results
        
        return state
