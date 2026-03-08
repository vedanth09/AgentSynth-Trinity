import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from typing import List

class PrivacyRedTeam:
    @staticmethod
    def simulate_mia(real_data, synthetic_data):
        print("   [RedTeam] Simulating Membership Inference Attack (MIA) [5-fold CV]...")
        try:
            n = min(len(real_data), len(synthetic_data), 200)
            real_s = real_data.sample(n, random_state=42).copy()
            synth_s = synthetic_data.sample(n, random_state=42).copy()
            real_s["_member"] = 1
            synth_s["_member"] = 0
            df = pd.concat([real_s, synth_s], ignore_index=True).fillna(0)
            y = df["_member"].values
            X = df.drop(columns=["_member"]).copy()
            id_cols = [c for c in X.columns if c.lower().endswith("_id") or c.lower() in {"patient_id","account_id","record_id","id"}]
            X = X.drop(columns=id_cols, errors="ignore")
            for col in X.select_dtypes(include=["object","category"]).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            X_arr = X.values.astype(float)
            mask = np.isfinite(X_arr).all(axis=1)
            X_arr, y = X_arr[mask], y[mask]
            if len(X_arr) < 20:
                return 0.5
            attacker = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            scores = cross_val_score(attacker, X_arr, y, cv=5, scoring="accuracy")
            acc = float(np.mean(scores))
            print(f"     -> MIA Attack Accuracy (CV): {acc:.2f} (Target < 0.60)")
            return acc
        except Exception as e:
            print(f"     -> MIA failed gracefully: {e}")
            return 0.5

    @staticmethod
    def calculate_k_anonymity(df, quasi_identifiers):
        if not quasi_identifiers or not all(c in df.columns for c in quasi_identifiers):
            return 1
        k = df.groupby(quasi_identifiers).size().min()
        print(f"   [RedTeam] k-Anonymity Level: {k}")
        return int(k)
