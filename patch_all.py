"""
Run from your AgentSynth-Trinity directory:
    python patch_all.py

Fixes:
  1. utils/metrics.py  — normalised Wasserstein, excludes patient_id
  2. app.py            — duplicate 'margin' keyword crash in correlation heatmaps
"""
import ast, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
#  FIX 1 — utils/metrics.py
# ══════════════════════════════════════════════════════════════════════════════
metrics_path = os.path.join(ROOT, "utils", "metrics.py")
if not os.path.exists(metrics_path):
    sys.exit(f"❌ Cannot find {metrics_path}")

with open(metrics_path) as f:
    m_src = f.read()

if "PATCHED_WASSERSTEIN_NORMALISED" in m_src:
    print("✅ Fix 1: metrics.py already patched")
else:
    OLD_M = '''def calculate_average_wasserstein_distance(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
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
        
    return float(np.mean(distances))'''

    NEW_M = '''# PATCHED_WASSERSTEIN_NORMALISED
def _is_id_col(col_name: str, values) -> bool:
    lower = col_name.lower()
    id_keywords = ("_id", "id_", "patient_id", "record_id", "index",
                   "uuid", "key", "rownum", "row_num", "seq")
    if lower == "id":
        return True
    if any(lower == k or lower.endswith(k) or lower.startswith(k)
           for k in id_keywords):
        return True
    import numpy as _np
    arr = _np.asarray(values)
    if len(arr) > 20 and (len(_np.unique(arr)) / len(arr)) > 0.95:
        return True
    return False


def calculate_average_wasserstein_distance(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Normalised average Wasserstein distance over numeric non-ID columns.
    Divides each column by its value range so all columns contribute equally.
    Excludes ID columns (patient_id, record_id, etc.) which would inflate
    the score to ~2059 even for a perfect synthesiser.
    """
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns
    distances = []

    if len(numeric_columns) == 0:
        return 0.0

    for col in numeric_columns:
        if col not in synthetic_data.columns:
            continue
        u_values = real_data[col].dropna().values
        v_values = synthetic_data[col].dropna().values
        if len(u_values) < 2 or len(v_values) < 2:
            continue
        if _is_id_col(col, u_values):
            continue
        col_range = float(u_values.max() - u_values.min())
        if col_range < 1e-6:
            continue
        dist = wasserstein_distance(u_values / col_range, v_values / col_range)
        distances.append(dist)

    if not distances:
        return 0.0

    return float(np.mean(distances))'''

    if OLD_M in m_src:
        m_patched = m_src.replace(OLD_M, NEW_M)
        try:
            ast.parse(m_patched)
        except SyntaxError as e:
            sys.exit(f"❌ metrics.py syntax error: {e}")
        with open(metrics_path, "w") as f:
            f.write(m_patched)
        print("✅ Fix 1: utils/metrics.py patched — normalised Wasserstein, patient_id excluded")
    else:
        print("⚠️  Fix 1: metrics.py pattern not matched — checking manually...")
        # Try a broader replacement: just insert the _is_id_col guard + normalisation
        # by rewriting the inner loop
        OLD_LOOP = '''    for col in numeric_columns:
        if col in synthetic_data.columns:
            u_values = real_data[col].dropna().values
            v_values = synthetic_data[col].dropna().values
            if len(u_values) > 0 and len(v_values) > 0:
                dist = wasserstein_distance(u_values, v_values)
                distances.append(dist)'''
        NEW_LOOP = '''    for col in numeric_columns:
        if col not in synthetic_data.columns:
            continue
        lower = col.lower()
        id_kw = ("_id","id_","patient_id","record_id","index","uuid","key","rownum","row_num","seq")
        if lower == "id" or any(lower == k or lower.endswith(k) or lower.startswith(k) for k in id_kw):
            continue
        u_values = real_data[col].dropna().values
        v_values = synthetic_data[col].dropna().values
        if len(u_values) < 2 or len(v_values) < 2:
            continue
        import numpy as _np2
        if len(u_values) > 20 and (len(_np2.unique(u_values)) / len(u_values)) > 0.95:
            continue
        col_range = float(u_values.max() - u_values.min())
        if col_range < 1e-6:
            continue
        dist = wasserstein_distance(u_values / col_range, v_values / col_range)
        distances.append(dist)'''
        if OLD_LOOP in m_src:
            m_patched = m_src.replace(OLD_LOOP, NEW_LOOP)
            try:
                ast.parse(m_patched)
            except SyntaxError as e:
                sys.exit(f"❌ metrics.py syntax error: {e}")
            with open(metrics_path, "w") as f:
                f.write(m_patched)
            print("✅ Fix 1: utils/metrics.py inner loop patched")
        else:
            print("❌ Fix 1: could not patch metrics.py — please show its current content")

# ══════════════════════════════════════════════════════════════════════════════
#  FIX 2 — app.py duplicate 'margin' in _chart_corr_heatmaps
# ══════════════════════════════════════════════════════════════════════════════
app_path = os.path.join(ROOT, "app.py")
if not os.path.exists(app_path):
    sys.exit(f"❌ Cannot find {app_path}")

with open(app_path) as f:
    a_src = f.read()

OLD_A = '''        fig.update_layout(**base, title_text=title, height=270,
                          margin=dict(l=30, r=20, t=36, b=30))'''

NEW_A = '''        # base already contains margin from _PL — strip it to avoid duplicate kwarg
        base_no_margin = {k: v for k, v in base.items() if k != "margin"}
        fig.update_layout(**base_no_margin, title_text=title, height=270,
                          margin=dict(l=30, r=20, t=36, b=30))'''

if "PATCHED_CORR_MARGIN" in a_src:
    print("✅ Fix 2: app.py margin already patched")
elif OLD_A in a_src:
    a_patched = a_src.replace(OLD_A, NEW_A + "\n        # PATCHED_CORR_MARGIN")
    try:
        ast.parse(a_patched)
    except SyntaxError as e:
        sys.exit(f"❌ app.py syntax error: {e}")
    with open(app_path, "w") as f:
        f.write(a_patched)
    print("✅ Fix 2: app.py duplicate margin keyword fixed")
else:
    print("⚠️  Fix 2: app.py pattern not found — may already be fixed")

print("\n🎉 Done. Run: streamlit run app.py")
