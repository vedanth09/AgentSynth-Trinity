"""
Run from AgentSynth-Trinity root:
    python debug_import.py

Tells us exactly which metrics.py Python is actually loading at runtime.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.metrics as m
import inspect

print("=== Module file loaded by Python ===")
print(f"  {inspect.getfile(m)}")

print("\n=== Is the fix present in the loaded module? ===")
src = inspect.getsource(m.calculate_average_wasserstein_distance)
print(f"  PATCHED: {'PATCHED_WASSERSTEIN' in src}")
print(f"  col_range / rng: {'rng' in src or 'col_range' in src}")
print(f"  _skip / is_id:   {'_skip' in src or 'is_id' in src}")

print("\n=== Quick live test ===")
import pandas as pd
import numpy as np

# Simulate: patient_id (should be excluded), age, treatment_cost
real = pd.DataFrame({
    'patient_id': range(1, 101),
    'age': np.random.randint(20, 80, 100),
    'treatment_cost': np.random.uniform(1000, 50000, 100),
})
synth = pd.DataFrame({
    'patient_id': range(1, 101),
    'age': np.random.randint(20, 80, 100),
    'treatment_cost': np.random.uniform(1000, 50000, 100),
})

score = m.calculate_average_wasserstein_distance(real, synth)
print(f"\n  Score with patient_id present: {score:.4f}")
print(f"  {'✅ FIXED — score is normalised (~0.01–0.15)' if score < 1.0 else '❌ STILL BROKEN — patient_id not excluded'}")
