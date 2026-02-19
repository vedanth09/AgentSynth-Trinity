import pandas as pd
from scipy.stats import wasserstein_distance
import numpy as np
from typing import List, Dict

def calculate_average_wasserstein_distance(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Calculates the average Wasserstein distance (earth mover's distance) 
    between numeric columns of real and synthetic datasets.

    Args:
        real_data (pd.DataFrame): The original real dataset.
        synthetic_data (pd.DataFrame): The generated synthetic dataset.

    Returns:
        float: The average Wasserstein distance across all numeric columns.
    """
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns
    distances = []

    if len(numeric_columns) == 0:
        return 0.0

    for col in numeric_columns:
        if col in synthetic_data.columns:
            # Drop NaNs for calculation
            u_values = real_data[col].dropna().values
            v_values = synthetic_data[col].dropna().values
            if len(u_values) > 0 and len(v_values) > 0:
                dist = wasserstein_distance(u_values, v_values)
                distances.append(dist)
    
    if not distances:
        return 0.0
        
    return float(np.mean(distances))
