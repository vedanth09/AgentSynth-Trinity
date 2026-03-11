import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, entropy
from typing import List, Dict, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

def _is_id_col(col_name: str, values: np.ndarray) -> bool:
    """
    Heuristic: return True if this column looks like a row identifier.
    ID columns blow up Wasserstein because SDV regenerates them as fresh
    integer sequences (1…N) whose absolute values differ from the real IDs.
    """
    lower = col_name.lower()
    # Name-based check
    id_keywords = ("_id", "id_", "patient_id", "record_id", "index",
                   "uuid", "key", "rownum", "row_num", "seq")
    if lower == "id" or any(lower == k or lower.endswith(k) or lower.startswith(k)
                            for k in id_keywords):
        return True
    # Uniqueness-ratio check: >95% unique values → likely a key column
    if len(values) > 20 and (len(np.unique(values)) / len(values)) > 0.95:
        return True
    return False


def calculate_average_wasserstein_distance(real_data, synthetic_data):
    # PATCHED_WASSERSTEIN v4
    ID_KW = ("_id","id_","patient_id","record_id","index","uuid","key","rownum","row_num","seq")

    def _skip(col, vals):
        lo = col.lower()
        if lo == "id" or any(lo == k or lo.endswith(k) or lo.startswith(k) for k in ID_KW):
            return True
        if len(vals) > 20 and (len(np.unique(vals)) / len(vals)) > 0.95:
            return True
        return False

    distances = []
    for col in real_data.select_dtypes(include=[np.number]).columns:
        if col not in synthetic_data.columns:
            continue
        u = real_data[col].dropna().values
        v = synthetic_data[col].dropna().values
        if len(u) < 2 or len(v) < 2 or _skip(col, u):
            continue
        rng = float(u.max() - u.min())
        if rng < 1e-6:
            continue
        distances.append(wasserstein_distance(u / rng, v / rng))

    return float(np.mean(distances)) if distances else 0.0

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

from sklearn.metrics.pairwise import rbf_kernel

def calculate_mmd(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Calculates Maximum Mean Discrepancy (MMD) with RBF kernel.
    
    Academic Motivation: MMD (Gretton et al., 2012) is a kernel-based test statistic 
    that measures the distance between distributions in a reproducing kernel 
    Hilbert space (RKHS).
    """
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0: return 0.0
    
    X = real_data[numeric_cols].dropna().values[:500]
    Y = synthetic_data[numeric_cols].dropna().values[:500]
    
    # Scrub Inf/NaN
    X = X[np.isfinite(X).all(axis=1)]
    Y = Y[np.isfinite(Y).all(axis=1)]
    
    if len(X) == 0 or len(Y) == 0: return 0.0
    
    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)
    
    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return float(np.sqrt(max(mmd, 0)))

def calculate_kl_divergence(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Calculates average KL-Divergence across all features.
    """
    divergences = []
    for col in real_data.columns:
        if col in synthetic_data.columns:
            p = real_data[col].value_counts(normalize=True).sort_index()
            q = synthetic_data[col].value_counts(normalize=True).sort_index()
            
            # Align
            all_idx = p.index.union(q.index)
            p_vals = np.array([p.get(i, 1e-10) for i in all_idx])
            q_vals = np.array([q.get(i, 1e-10) for i in all_idx])
            
            divergences.append(entropy(p_vals, q_vals))
            
    return float(np.mean(divergences)) if divergences else 0.0

def train_test_utility_evaluation(real_data: pd.DataFrame, 
                                  synthetic_data: pd.DataFrame, 
                                  target_col: str) -> Dict[str, float]:
    """
    Advanced Utility: TSTR (Train-Synthetic-Test-Real) and TRTS (Train-Real-Test-Synthetic).
    
    Academic Motivation: The TSTR/TRTS accuracy gap (Jordon et al., 2018) measures 
    preservation of decision boundaries across the real-synthetic divide.
    """
    if target_col not in real_data.columns or target_col not in synthetic_data.columns:
        return {"performance_drop": 0.0}

    def preprocess(df):
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object']):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        return df_encoded

    try:
        from xgboost import XGBClassifier  # lazy import — avoids libomp crash at module load
        real_proc = preprocess(real_data.dropna())
        synth_proc = preprocess(synthetic_data.dropna())
        
        # Label Encode the Target Column specifically for XGBoost compatibility
        le_target = LabelEncoder()
        
        X_real = real_proc.drop(columns=[target_col])
        y_real = pd.Series(le_target.fit_transform(real_proc[target_col]))
        
        X_synth = synth_proc.drop(columns=[target_col])
        # Handle unseen labels in synthetic data by falling back to -1 or similar, but
        # since TRTS trains on synthetic and tests on real, it's safer to fit a new encoder
        # per dataset or fit on the union. Let's fit on the union to be safe.
        le_target.fit(pd.concat([real_proc[target_col], synth_proc[target_col]]))
        
        y_real = pd.Series(le_target.transform(real_proc[target_col]))
        y_synth = pd.Series(le_target.transform(synth_proc[target_col]))

        # Scrub Inf
        real_mask = np.isfinite(X_real.select_dtypes(include=[np.number])).all(axis=1)
        X_real, y_real = X_real[real_mask], y_real[real_mask]
        
        synth_mask = np.isfinite(X_synth.select_dtypes(include=[np.number])).all(axis=1)
        X_synth, y_synth = X_synth[synth_mask], y_synth[synth_mask]
        
        if len(X_real) < 10 or len(X_synth) < 10:
            return {"performance_drop": 0.0}
        
        X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_real, y_real, test_size=0.3)
        
        # 1. Baseline (R -> R)
        model_rr = XGBClassifier().fit(X_r_train, y_r_train)
        acc_rr = accuracy_score(y_r_test, model_rr.predict(X_r_test))
        
        # 2. TSTR (S -> R)
        model_sr = XGBClassifier().fit(X_synth, y_synth)
        acc_sr = accuracy_score(y_r_test, model_sr.predict(X_r_test))
        
        # 3. TRTS (R -> S)
        acc_rs = accuracy_score(y_synth, model_rr.predict(X_synth))
        
        return {
            "baseline_accuracy": float(acc_rr),
            "tstr_accuracy": float(acc_sr),
            "trts_accuracy": float(acc_rs),
            "performance_drop": float(max(0, acc_rr - acc_sr)) * 100
        }
    except Exception as e:
        print(f"Utility Error: {e}")
        return {"performance_drop": 100.0}

def check_linkability(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
    """Privacy Audit: Exact matches."""
    real_set = set(map(tuple, real_data.values))
    synth_set = set(map(tuple, synthetic_data.values))
    matches = len(real_set.intersection(synth_set))
    return {"exact_matches": matches, "is_compliant": matches == 0}
