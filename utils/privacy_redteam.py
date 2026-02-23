import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class PrivacyRedTeam:
    """
    Utility for automated red-team testing of synthetic data privacy.
    
    Academic Motivation: Membership Inference Attacks (MIA) (Choquette-Chooz et al., 
    2021) serve as an empirical 'red-team' test to validate whether a generative 
    model has 'memorized' specific training records.
    """
    
    @staticmethod
    def simulate_mia(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """
        Simulates a Membership Inference Attack.
        Returns the attack accuracy (risk score). 0.5 is random guess (strong privacy).
        """
        print("   [RedTeam] Simulating Membership Inference Attack (MIA)...")
        
        # 1. Prepare Attack Dataset
        # "In" members (real data used for training)
        real_sample = real_data.sample(min(len(real_data), 100))
        real_sample['is_member'] = 1
        
        # "Out" members (synthetic data)
        synth_sample = synthetic_data.sample(min(len(synthetic_data), 100))
        synth_sample['is_member'] = 0
        
        attack_df = pd.concat([real_sample, synth_sample]).fillna(0)
        X = attack_df.drop('is_member', axis=1)
        y = attack_df['is_member']
        
        # Use numerical columns only for simple attack model
        X_num = X.select_dtypes(include=[np.number])
        
        if X_num.shape[1] == 0:
            return 0.5 # Cannot attack if no numerical data
            
        X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.3)
        
        # 2. Train Shadow Model (Attacker)
        attacker = RandomForestClassifier(n_estimators=50, max_depth=5)
        attacker.fit(X_train, y_train)
        
        # 3. Evaluate Attack Success
        preds = attacker.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"     -> MIA Attack Accuracy: {acc:.2f} (Target < 0.60)")
        return acc

    @staticmethod
    def calculate_k_anonymity(df: pd.DataFrame, quasi_identifiers: List[str]) -> int:
        """
        Calculates k-anonymity for given quasi-identifiers.
        
        Academic Motivation: k-anonymity (Sweeney, 2002) ensures that each record 
        is indistinguishable from at least k-1 other records in the release.
        """
        if not quasi_identifiers or not all(c in df.columns for c in quasi_identifiers):
            return 1
            
        # Group by quasi-identifiers and find the smallest group size
        k = df.groupby(quasi_identifiers).size().min()
        print(f"   [RedTeam] k-Anonymity Level: {k}")
        return int(k)
