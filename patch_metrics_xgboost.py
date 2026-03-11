"""
Run from AgentSynth-Trinity root:
    python patch_metrics_xgboost.py

Moves the XGBoost import inside the function that uses it so that
metrics.py loads cleanly even when libomp is missing.
"""
import os, sys, ast

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "metrics.py")
with open(target) as f:
    src = f.read()

# 1. Remove top-level XGBoost import
OLD_IMPORT = "from xgboost import XGBClassifier\n"
if OLD_IMPORT in src:
    src = src.replace(OLD_IMPORT, "")
    print("✅ Removed top-level XGBoost import")
else:
    print("ℹ️  Top-level XGBoost import already removed")

# 2. Add lazy import inside train_test_utility_evaluation
OLD_FUNC_START = "def train_test_utility_evaluation("
OLD_TRY = "    try:\n        real_proc = preprocess(real_data.dropna())"
NEW_TRY = "    try:\n        from xgboost import XGBClassifier  # lazy import — avoids libomp crash at module load\n        real_proc = preprocess(real_data.dropna())"

if "from xgboost import XGBClassifier  # lazy" in src:
    print("ℹ️  Lazy XGBoost import already present")
elif OLD_TRY in src:
    src = src.replace(OLD_TRY, NEW_TRY)
    print("✅ Added lazy XGBoost import inside train_test_utility_evaluation")
else:
    print("⚠️  Could not find try block — inserting at function start")
    idx = src.find("def train_test_utility_evaluation(")
    idx = src.find("\n    try:", idx)
    if idx > 0:
        src = src[:idx+5] + "\n        from xgboost import XGBClassifier  # lazy" + src[idx+5:]
        print("✅ Inserted lazy import")

try:
    ast.parse(src)
    print("✅ Syntax OK")
except SyntaxError as e:
    sys.exit(f"❌ {e}")

with open(target, "w") as f:
    f.write(src)
print("✅ utils/metrics.py written")

# 3. Quick import test
print("\n=== Testing import ===")
import subprocess, json
result = subprocess.run(
    ["/opt/homebrew/bin/python3.11", "-c",
     "import sys; sys.path.insert(0,'.'); from utils.metrics import calculate_average_wasserstein_distance; "
     "import numpy as np, pandas as pd; "
     "r=pd.DataFrame({'patient_id':range(100),'age':np.random.randint(20,80,100),'treatment_cost':np.random.uniform(1000,50000,100)}); "
     "s=pd.DataFrame({'patient_id':range(100),'age':np.random.randint(20,80,100),'treatment_cost':np.random.uniform(1000,50000,100)}); "
     "print('Score:', calculate_average_wasserstein_distance(r,s))"],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(result.stdout.strip())
    score_str = result.stdout.split("Score:")[-1].strip()
    score = float(score_str)
    if score < 1.0:
        print(f"✅ FIXED — normalised score {score:.4f} (was 2059)")
    else:
        print(f"❌ Still broken — score {score:.4f}")
else:
    print("Import error:")
    print(result.stderr[-800:])
