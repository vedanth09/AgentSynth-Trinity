import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, entropy
from typing import List, Dict, Optional, Tuple, Any
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

def calculate_average_wasserstein_distance(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Calculates the average Wasserstein distance (earth mover's distance) 
    between numeric columns of real and synthetic datasets.
    """
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns
    distances = []

    if len(numeric_columns) == 0:
        return 0.0

    for col in numeric_columns:
        if col in synthetic_data.columns:
            u_values = real_data[col].dropna().values
            v_values = synthetic_data[col].dropna().values
            if len(u_values) > 0 and len(v_values) > 0:
                dist = wasserstein_distance(u_values, v_values)
                distances.append(dist)
    
    if not distances:
        return 0.0
        
    return float(np.mean(distances))

def calculate_jensen_shannon_divergence(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Calculates average Jensen-Shannon Divergence for categorical columns.
    Lower is better (0 = identical distributions).
    """
    cat_columns = real_data.select_dtypes(include=['object', 'category']).columns
    divergences = []

    if len(cat_columns) == 0:
        return 0.0

    for col in cat_columns:
        if col in synthetic_data.columns:
            # Get probability distributions
            real_probs = real_data[col].value_counts(normalize=True)
            synth_probs = synthetic_data[col].value_counts(normalize=True)
            
            # Align indices (union of categories)
            all_cats = set(real_probs.index).union(set(synth_probs.index))
            p = np.array([real_probs.get(c, 0) for c in all_cats])
            q = np.array([synth_probs.get(c, 0) for c in all_cats])
            
            # Calculate JS Divergence
            m = 0.5 * (p + q)
            js_div = 0.5 * (entropy(p, m) + entropy(q, m))
            divergences.append(js_div)

    if not divergences:
        return 0.0
        
    return float(np.nanmean(divergences))

def calculate_correlation_similarity(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Compares correlation matrices. Returns 1 - L2 norm of the difference.
    Higher is better (1.0 = identical correlations).
    """
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return 1.0

    real_corr = real_data[numeric_cols].corr().fillna(0)
    synth_corr = synthetic_data[numeric_cols].corr().fillna(0)
    
    diff = np.linalg.norm(real_corr.values - synth_corr.values)
    # Normalize by size to keep reasonable scale, simpler approach:
    # return difference magnitude. Let's return 1 / (1 + diff) for a 0-1 score roughly
    return 1.0 / (1.0 + diff)

def train_test_utility_evaluation(real_data: pd.DataFrame, 
                                  synthetic_data: pd.DataFrame, 
                                  target_col: str) -> Dict[str, float]:
    """
    TSTR (Train-Synthetic-Test-Real) Evaluation using XGBoost.
    Trains on Synthetic, Tests on Real.
    Returns Accuracy and F1 Score relative to Real-Real Baseline.
    """
    if target_col not in real_data.columns or target_col not in synthetic_data.columns:
        return {"accuracy_drop": 0.0, "f1_drop": 0.0}

    # Preprocessing (Simple Label Encoding for simplicity in this prototype)
    def preprocess(df):
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object']):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        return df_encoded

    try:
        real_proc = preprocess(real_data.dropna())
        synth_proc = preprocess(synthetic_data.dropna())
        
        X_real = real_proc.drop(columns=[target_col])
        y_real = real_proc[target_col]
        
        X_synth = synth_proc.drop(columns=[target_col])
        y_synth = synth_proc[target_col]
        
        # 1. Baseline: Train Real, Test Real
        X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
        model_baseline = XGBClassifier(eval_metric='logloss')
        model_baseline.fit(X_train, y_train)
        preds_baseline = model_baseline.predict(X_test)
        acc_baseline = accuracy_score(y_test, preds_baseline)
        
        # 2. TSTR: Train Synthetic, Test Real
        model_tstr = XGBClassifier(eval_metric='logloss')
        model_tstr.fit(X_synth, y_synth)
        preds_tstr = model_tstr.predict(X_test) # Test on Real Holdout
        acc_tstr = accuracy_score(y_test, preds_tstr)
        
        drop = (acc_baseline - acc_tstr) / acc_baseline if acc_baseline > 0 else 0.0
        
        return {
            "baseline_accuracy": float(acc_baseline),
            "tstr_accuracy": float(acc_tstr),
            "performance_drop": float(drop) * 100 # Percentage
        }
    except Exception as e:
        print(f"Utility Eval Failed: {e}")
        return {"performance_drop": 100.0}

def check_linkability(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks for exact row matches between real and synthetic data (Privacy Audit).
    """
    # Simply check intersections
    # Convert to tuples for hashable comparison
    real_set = set(map(tuple, real_data.values))
    synth_set = set(map(tuple, synthetic_data.values))
    
    exact_matches = real_set.intersection(synth_set)
    count = len(exact_matches)
    
    return {
        "exact_matches": count,
        "is_compliant": count == 0
    }
