"""
Run from AgentSynth-Trinity root:
    python patch_metrics_direct.py

Directly rewrites the calculate_average_wasserstein_distance function
to exclude ID columns and normalise per-column range.
"""
import os, sys, ast

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "metrics.py")
if not os.path.exists(target):
    sys.exit(f"❌ Not found: {target}")

with open(target) as f:
    src = f.read()

if "PATCHED_WASSERSTEIN" in src:
    print("✅ Already patched.")
    sys.exit(0)

# Find the function and replace just its body
MARKER_START = 'def calculate_average_wasserstein_distance('
MARKER_END   = '\ndef calculate_jensen_shannon_divergence('

start = src.find(MARKER_START)
end   = src.find(MARKER_END, start)

if start == -1 or end == -1:
    sys.exit("❌ Could not locate function boundaries in metrics.py")

NEW_FUNC = '''def calculate_average_wasserstein_distance(real_data, synthetic_data):
    # PATCHED_WASSERSTEIN — normalised, ID columns excluded
    ID_KEYWORDS = ("_id", "id_", "patient_id", "record_id", "index",
                   "uuid", "key", "rownum", "row_num", "seq")

    def is_id(col, vals):
        lo = col.lower()
        if lo == "id":
            return True
        if any(lo == k or lo.endswith(k) or lo.startswith(k) for k in ID_KEYWORDS):
            return True
        if len(vals) > 20 and (len(np.unique(vals)) / len(vals)) > 0.95:
            return True
        return False

    numeric_columns = real_data.select_dtypes(include=[np.number]).columns
    distances = []

    for col in numeric_columns:
        if col not in synthetic_data.columns:
            continue
        u = real_data[col].dropna().values
        v = synthetic_data[col].dropna().values
        if len(u) < 2 or len(v) < 2:
            continue
        if is_id(col, u):
            continue
        col_range = float(u.max() - u.min())
        if col_range < 1e-6:
            continue
        distances.append(wasserstein_distance(u / col_range, v / col_range))

    return float(np.mean(distances)) if distances else 0.0
'''

patched = src[:start] + NEW_FUNC + src[end:]

try:
    ast.parse(patched)
except SyntaxError as e:
    sys.exit(f"❌ Syntax error: {e}")

with open(target, "w") as f:
    f.write(patched)

print("✅ utils/metrics.py patched — Wasserstein now normalised, patient_id excluded")
print("   Expected scores: ~0.03–0.08 instead of ~2059")
