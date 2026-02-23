import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List

class TrinityExplainer:
    """
    XAI Utility for explaining synthetic data utility.
    
    Academic Motivation: SHAP (Lundberg & Lee, 2017) provides a game-theoretic 
    approach to explain the output of machine learning models, essential for 
    human-in-the-loop oversight in regulated domains.
    """
    
    @staticmethod
    def explain_utility(real_data: pd.DataFrame, 
                        synthetic_data: pd.DataFrame, 
                        target_col: str):
        """
        Uses SHAP to identify feature importance in the utility model. 
        Helps auditors understand which features drive the similarity.
        """
        print(f"   [Explainability] Running SHAP analysis on target: {target_col}")
        
        # Prepare data (subset for speed)
        df = real_data.dropna().sample(min(len(real_data), 200))
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = df[target_col]
        
        if X.shape[1] == 0:
            print("     -> No numerical features for SHAP.")
            return None

        # Train model
        model = XGBClassifier().fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        # Aggregate importance
        feature_importance = np.abs(shap_values.values).mean(0)
        importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
        importance_df = importance_df.sort_values(by='importance', ascending=False)
        
        top_3 = importance_df.head(3)['feature'].tolist()
        print(f"     -> Top 3 Primary Utility Drivers: {top_3}")
        return importance_df
