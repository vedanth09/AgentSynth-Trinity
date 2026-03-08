# Run this script from your AgentSynth-Trinity directory:
# python patch_metrics.py
# It surgically patches utils/metrics.py to exclude patient_id from Wasserstein.

import re

with open("utils/metrics.py", "r") as f:
    content = f.read()

# Find the calculate_average_wasserstein_distance function and patch it
old = '''def calculate_average_wasserstein_distance(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Calculates the average NORMALISED Wasserstein distance over numeric columns.
    Each column is scaled to [0,1] before comparison so high-magnitude columns
    like treatment_cost don\'t dominate the score.
    """
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns'''

new = '''def calculate_average_wasserstein_distance(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Calculates the average NORMALISED Wasserstein distance over numeric columns.
    Each column is scaled to [0,1] before comparison so high-magnitude columns
    like treatment_cost don\'t dominate the score.
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
                       if not _is_id_col(c, real_data[c])]'''

if old in content:
    content = content.replace(old, new)
    with open("utils/metrics.py", "w") as f:
        f.write(content)
    print("✅ Patched successfully! patient_id will be excluded from Wasserstein.")
else:
    # Try a simpler approach - just patch the numeric_columns line
    old2 = "    numeric_columns = real_data.select_dtypes(include=[np.number]).columns\n    distances = []"
    new2 = """    # Exclude ID/surrogate-key columns — huge ranges corrupt the average
    def _is_id_col(col, series):
        if col.lower() in {"patient_id","account_id","record_id","user_id","id"} or col.lower().endswith("_id"):
            return True
        if series.nunique() > 0.9 * len(series) and pd.api.types.is_numeric_dtype(series):
            return True
        return False
    numeric_columns = [c for c in real_data.select_dtypes(include=[np.number]).columns
                       if not _is_id_col(c, real_data[c])]
    distances = []"""
    
    if old2 in content:
        content = content.replace(old2, new2)
        with open("utils/metrics.py", "w") as f:
            f.write(content)
        print("✅ Patched successfully (method 2)!")
    else:
        print("❌ Could not auto-patch. Please manually add ID exclusion.")
        print()
        print("Find this line in utils/metrics.py:")
        print("    numeric_columns = real_data.select_dtypes(include=[np.number]).columns")
        print()
        print("Replace it with:")
        print("""    def _is_id_col(col, series):
        if col.lower() in {"patient_id","account_id","record_id","user_id","id"} or col.lower().endswith("_id"):
            return True
        if series.nunique() > 0.9 * len(series) and pd.api.types.is_numeric_dtype(series):
            return True
        return False
    numeric_columns = [c for c in real_data.select_dtypes(include=[np.number]).columns
                       if not _is_id_col(c, real_data[c])]""")

# Also check privacy_redteam.py
with open("utils/privacy_redteam.py", "r") as f:
    redteam = f.read()
if "cross_val_score" in redteam:
    print("✅ privacy_redteam.py already has cross_val_score fix.")
else:
    print("⚠️  privacy_redteam.py still needs updating — cross_val_score missing.")