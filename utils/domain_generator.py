import pandas as pd
import numpy as np
from typing import Optional


class DomainDataGenerator:
    DIAGNOSES = [
        "Hypertension", "Type 2 Diabetes", "Coronary Artery Disease",
        "Asthma", "Osteoarthritis", "Chronic Kidney Disease",
        "Heart Failure", "COPD", "Depression", "Lupus",
        "Rheumatoid Arthritis", "Hypothyroidism", "Obesity",
        "Atrial Fibrillation", "Stroke",
    ]
    DIAGNOSIS_PRIORS = {
        "Hypertension":             (58, 12, 250,   120, 148),
        "Type 2 Diabetes":          (52, 13, 380,   180, 128),
        "Coronary Artery Disease":  (63, 10, 5200, 2000, 135),
        "Asthma":                   (35, 18, 175,    80, 118),
        "Osteoarthritis":           (64, 11, 750,   300, 125),
        "Chronic Kidney Disease":   (61, 12, 3800, 1500, 132),
        "Heart Failure":            (68, 10, 7500, 3000, 138),
        "COPD":                     (64, 11, 1200,  500, 126),
        "Depression":               (40, 15, 300,   150, 118),
        "Lupus":                    (36, 12, 2200, 1000, 122),
        "Rheumatoid Arthritis":     (52, 13, 1800,  800, 124),
        "Hypothyroidism":           (48, 15, 125,    60, 120),
        "Obesity":                  (42, 14, 450,   200, 130),
        "Atrial Fibrillation":      (66, 10, 3000, 1200, 135),
        "Stroke":                   (67, 11, 12000, 4000, 142),
    }
    TRANSACTION_TYPES = ["Purchase","Withdrawal","Transfer","Payment","Refund","Deposit","Fee"]
    MERCHANT_CATEGORIES = ["Grocery","Restaurant","Travel","Healthcare","Entertainment","Retail","Utilities","Online"]

    def generate_healthcare(self, n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        rows = []
        weights = np.array([46,39,30,27,34,27,36,35,31,29,38,36,27,24,41], dtype=float)
        weights /= weights.sum()
        for i in range(n_rows):
            diagnosis = np.random.choice(self.DIAGNOSES, p=weights)
            age_mu, age_sd, cost_mu, cost_sd, bp_mu = self.DIAGNOSIS_PRIORS[diagnosis]
            age  = int(np.clip(np.random.normal(age_mu, age_sd), 18, 95))
            cost = round(max(20.0, np.random.normal(cost_mu, cost_sd)), 2)
            bp   = int(np.clip(np.random.normal(bp_mu, 14), 85, 200))
            bmi  = round(np.clip(np.random.normal(27.5, 5.5), 15.0, 55.0), 1)
            gender     = np.random.choice(["Male", "Female"], p=[0.48, 0.52])
            smoker     = int(np.random.choice([0, 1], p=[0.75, 0.25]))
            num_meds   = int(np.clip(np.random.poisson(3.5), 0, 15))
            readmitted = int((cost > 3000 or age > 70) and np.random.rand() > 0.6)
            rows.append({
                "patient_id": 1000 + i + 1, "age": age, "gender": gender,
                "bmi": bmi, "systolic_bp": bp, "smoker": smoker,
                "diagnosis": diagnosis, "num_medications": num_meds,
                "treatment_cost": cost, "readmitted": readmitted,
            })
        df = pd.DataFrame(rows)
        print(f"[DomainGenerator] Generated {len(df)} healthcare records from domain priors.")
        return df

    def generate_finance(self, n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        rows = []
        for i in range(n_rows):
            tx_type  = np.random.choice(self.TRANSACTION_TYPES, p=[0.35,0.15,0.15,0.15,0.05,0.10,0.05])
            merchant = np.random.choice(self.MERCHANT_CATEGORIES)
            is_fraud = int(np.random.rand() < 0.05)
            if is_fraud:
                amount = round(np.random.uniform(500, 8000), 2)
                hour   = int(np.random.choice([1, 2, 3, 23, 0]))
            else:
                amount = round(min(np.random.lognormal(4.0, 1.2), 5000), 2)
                hour   = int(np.clip(np.random.normal(13, 4), 0, 23))
            rows.append({
                "transaction_id": 100000 + i + 1,
                "customer_age": int(np.clip(np.random.normal(42, 14), 18, 85)),
                "transaction_type": tx_type, "merchant_category": merchant,
                "amount": amount, "hour_of_day": hour,
                "account_balance": round(np.random.lognormal(8.5, 1.5), 2),
                "num_tx_today": int(np.clip(np.random.poisson(3), 1, 25)),
                "credit_score": int(np.clip(np.random.normal(680, 80), 300, 850)),
                "foreign_transaction": int(np.random.rand() < 0.08),
                "is_fraud": is_fraud,
            })
        df = pd.DataFrame(rows)
        print(f"[DomainGenerator] Generated {len(df)} finance records from domain priors.")
        return df

    def generate(self, domain: str, n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
        if domain.lower() == "healthcare":
            return self.generate_healthcare(n_rows, seed)
        elif domain.lower() == "finance":
            return self.generate_finance(n_rows, seed)
        else:
            raise ValueError(f"Unknown domain: {domain}")
