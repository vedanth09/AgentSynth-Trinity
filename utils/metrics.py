import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, entropy
from typing import List, Dict, Optional, Tuple, Any
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import rbf_kernel


def calculate_average_wasserstein_distance(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Calculates the average NORMALISED Wasserstein distance over numeric columns.
    Each column is scaled to [0,1] before comparison so high-magnitude columns
    like treatment_cost don't dominate the score.
    ID columns (patient_id etc.) are excluded as their large ranges corrupt scores.
    """
    # Exclude ID/surrogate-key columns — they have huge ranges that dominate the average
    def _is_id_col(col, series):
        if col.lower() in {"patient_id","account_id","record_id","user_id","id"} or col.lower().endswith("_id"):
            return True
        if series.nunique() > 0.9 * len(series) and pd.api.types.is_numeric_dtype(series):
            return True
        return False

    numeric_columns = [c for c in real_data.select_dtypes(include=[np.number]).columns
                       if not _is_id_col(c, real_data[c])]
    distances = []

    if len(numeric_columns) == 0:
        return 0.0

    for col in numeric_columns:
        if col in synthetic_data.columns:
            u_values = real_data[col].dropna().values.astype(float)
            v_values = synthetic_data[col].dropna().values.astype(float)
            if len(u_values) > 0 and len(v_values) > 0:
                # Normalise to [0,1] so all columns contribute equally
                col_min = min(u_values.min(), v_values.min())
                col_max = max(u_values.max(), v_values.max())
                col_range = max(col_max - col_min, 1e-6)
                u_norm = (u_values - col_min) / col_range
                v_norm = (v_values - col_min) / col_range
                distances.append(wasserstein_distance(u_norm, v_norm))

    return float(np.mean(distances)) if distances else 0.0


def calculate_jensen_shannon_divergence(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """Average Jensen-Shannon Divergence for categorical columns. Lower = better."""
    cat_columns = real_data.select_dtypes(include=['object', 'category']).columns
    divergences = []

    if len(cat_columns) == 0:
        return 0.0

    for col in cat_columns:
        if col in synthetic_data.columns:
            real_probs  = real_data[col].value_counts(normalize=True)
            synth_probs = synthetic_data[col].value_counts(normalize=True)
            all_cats = set(real_probs.index).union(set(synth_probs.index))
            p = np.array([real_probs.get(c, 0) for c in all_cats])
            q = np.array([synth_probs.get(c, 0) for c in all_cats])
            m = 0.5 * (p + q)
            js_div = 0.5 * (entropy(p + 1e-10, m + 1e-10) + entropy(q + 1e-10, m + 1e-10))
            divergences.append(js_div)

    return float(np.nanmean(divergences)) if divergences else 0.0


def calculate_correlation_similarity(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """Compares correlation matrices. Higher = better (1.0 = identical)."""
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return 1.0
    real_corr  = real_data[numeric_cols].corr().fillna(0)
    synth_corr = synthetic_data[numeric_cols].corr().fillna(0)
    diff = np.linalg.norm(real_corr.values - synth_corr.values)
    return 1.0 / (1.0 + diff)


def calculate_mmd(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """Maximum Mean Discrepancy with RBF kernel (Gretton et al., 2012)."""
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 0.0
    X = real_data[numeric_cols].dropna().values[:500].astype(float)
    Y = synthetic_data[numeric_cols].dropna().values[:500].astype(float)
    X = X[np.isfinite(X).all(axis=1)]
    Y = Y[np.isfinite(Y).all(axis=1)]
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)
    return float(np.sqrt(max(K_XX.mean() + K_YY.mean() - 2 * K_XY.mean(), 0)))


def calculate_kl_divergence(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """Average KL-Divergence across all features."""
    divergences = []
    for col in real_data.columns:
        if col in synthetic_data.columns:
            p = real_data[col].value_counts(normalize=True).sort_index()
            q = synthetic_data[col].value_counts(normalize=True).sort_index()
            all_idx = p.index.union(q.index)
            p_vals = np.array([p.get(i, 1e-10) for i in all_idx])
            q_vals = np.array([q.get(i, 1e-10) for i in all_idx])
            divergences.append(entropy(p_vals, q_vals))
    return float(np.mean(divergences)) if divergences else 0.0


def _safe_encode_target(real_series: pd.Series, synth_series: pd.Series):
    """
    Fits a LabelEncoder on the union of real + synthetic target values,
    then transforms both. Returns (y_real_encoded, y_synth_encoded).
    Handles type mismatches (int vs string '0'/'1') by casting to str first.
    """
    le = LabelEncoder()
    combined = pd.concat([real_series, synth_series]).astype(str)
    le.fit(combined)
    return (
        le.transform(real_series.astype(str)),
        le.transform(synth_series.astype(str)),
    )


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns so XGBoost can consume them."""
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].astype(str))
    return out


def train_test_utility_evaluation(real_data: pd.DataFrame,
                                  synthetic_data: pd.DataFrame,
                                  target_col: str) -> Dict[str, float]:
    """
    TSTR / TRTS utility evaluation (Jordon et al., 2018).

    Train-Synthetic-Test-Real (TSTR): model trained on synthetic, tested on real.
    Train-Real-Test-Synthetic (TRTS): model trained on real, tested on synthetic.
    The TSTR accuracy gap measures preservation of decision boundaries.
    """
    if target_col not in real_data.columns or target_col not in synthetic_data.columns:
        return {"performance_drop": 0.0}

    try:
        # Drop NAs and reset index to guarantee clean 0-based indexing
        real_clean  = real_data.dropna().reset_index(drop=True)
        synth_clean = synthetic_data.dropna().reset_index(drop=True)

        # Encode features
        real_enc  = _encode_features(real_clean)
        synth_enc = _encode_features(synth_clean)

        # Encode target — fit on union to avoid unseen-label errors
        y_real_arr, y_synth_arr = _safe_encode_target(
            real_enc[target_col], synth_enc[target_col]
        )

        X_real  = real_enc.drop(columns=[target_col]).values
        X_synth = synth_enc.drop(columns=[target_col]).values

        # Remove non-finite rows using numpy masks (no index alignment issues)
        real_mask  = np.isfinite(X_real).all(axis=1)
        synth_mask = np.isfinite(X_synth).all(axis=1)

        X_real,  y_real  = X_real[real_mask],   y_real_arr[real_mask]
        X_synth, y_synth = X_synth[synth_mask], y_synth_arr[synth_mask]

        if len(X_real) < 10 or len(X_synth) < 10:
            print(f"   [Utility] Too few rows after cleaning, skipping.")
            return {"performance_drop": 0.0, "tstr_accuracy": 0.0,
                    "trts_accuracy": 0.0, "baseline_accuracy": 0.0}

        X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42
        )

        # Baseline R→R
        model_rr = XGBClassifier(eval_metric="logloss", verbosity=0)
        model_rr.fit(X_r_train, y_r_train)
        acc_rr = accuracy_score(y_r_test, model_rr.predict(X_r_test))

        # TSTR S→R
        model_sr = XGBClassifier(eval_metric="logloss", verbosity=0)
        model_sr.fit(X_synth, y_synth)
        acc_sr = accuracy_score(y_r_test, model_sr.predict(X_r_test))

        # TRTS R→S
        acc_rs = accuracy_score(y_synth, model_rr.predict(X_synth))

        return {
            "baseline_accuracy": float(acc_rr),
            "tstr_accuracy":     float(acc_sr),
            "trts_accuracy":     float(acc_rs),
            "performance_drop":  float(max(0, acc_rr - acc_sr)) * 100,
        }

    except Exception as e:
        print(f"Utility Error: {e}")
        return {"performance_drop": 0.0, "tstr_accuracy": 0.0,
                "trts_accuracy": 0.0, "baseline_accuracy": 0.0}


def check_linkability(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
    """Privacy audit: checks for exact row matches between real and synthetic data."""
    real_set  = set(map(tuple, real_data.astype(str).values))
    synth_set = set(map(tuple, synthetic_data.astype(str).values))
    matches   = len(real_set.intersection(synth_set))
    return {"exact_matches": matches, "is_compliant": matches == 0}