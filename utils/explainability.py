import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List, Optional

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
                        target_col: str) -> Optional[pd.DataFrame]:
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

        if len(X) < 5:
            print("     -> Too few samples for SHAP analysis.")
            return None

        # FIX: XGBClassifier requires binary/integer class labels (0, 1, 2...).
        # If the target is continuous (e.g. age=25,30) or has too many unique
        # values, binarize it at the median so XGBoost gets a valid classification target.
        if y.dtype in [np.float64, np.float32] or y.nunique() > 10:
            median_val = y.median()
            y = (y > median_val).astype(int)
            print(f"     -> Continuous target binarized at median ({median_val}) for SHAP model.")
        elif y.dtype == object:
            # Label-encode string targets
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)))

        # Guard: need at least 2 classes to train a classifier
        if y.nunique() < 2:
            print("     -> Target has only one class after binarization, skipping SHAP.")
            return None

        try:
            model = XGBClassifier(eval_metric="logloss", verbosity=0).fit(X, y)
            
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            
            # For multi-output SHAP (multi-class), take mean across classes
            sv = shap_values.values
            if sv.ndim == 3:
                sv = sv.mean(axis=2)

            feature_importance = np.abs(sv).mean(0)
            importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
            importance_df = importance_df.sort_values(by='importance', ascending=False)
            
            top_3 = importance_df.head(3)['feature'].tolist()
            print(f"     -> Top 3 Primary Utility Drivers: {top_3}")
            return importance_df

        except Exception as e:
            print(f"     -> SHAP analysis failed gracefully: {e}")
            return None