"""
Run from AgentSynth-Trinity root:
    python verify_and_force_patch.py
"""
import os, sys

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "metrics.py")

with open(target) as f:
    src = f.read()

print("=== First 35 lines of utils/metrics.py ===")
for i, line in enumerate(src.splitlines()[:35], 1):
    print(f"{i:3d}: {line}")

print("\n=== Is patched? ===")
print(f"  PATCHED_WASSERSTEIN in file: {'PATCHED_WASSERSTEIN' in src}")
print(f"  _is_id_col in file:          {'_is_id_col' in src or 'is_id' in src}")
print(f"  col_range in file:           {'col_range' in src}")

# Force-write the fixed function regardless
print("\n=== Force-writing fixed function ===")

MARKER_START = 'def calculate_average_wasserstein_distance('
MARKER_END   = '\ndef calculate_jensen_shannon_divergence('

start = src.find(MARKER_START)
end   = src.find(MARKER_END, start)
print(f"  Function found at chars {start}–{end}")

NEW_FUNC = '''def calculate_average_wasserstein_distance(real_data, synthetic_data):
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
'''

patched = src[:start] + NEW_FUNC + src[end:]

import ast
try:
    ast.parse(patched)
    print("  Syntax OK")
except SyntaxError as e:
    sys.exit(f"  ❌ Syntax error: {e}")

with open(target, "w") as f:
    f.write(patched)
print("  ✅ Written successfully")

# Verify it's actually on disk now
with open(target) as f:
    check = f.read()
print(f"  Verified on disk: {'PATCHED_WASSERSTEIN v4' in check}")
