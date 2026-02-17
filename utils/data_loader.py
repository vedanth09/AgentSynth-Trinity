import pandas as pd
import os
from typing import Optional

class DataLoader:
    """
    Utility class for loading datasets in the AgentSynth-Trinity framework.
    """

    @staticmethod
    def load_csv(file_path: str) -> Optional[pd.DataFrame]:
        """
        Loads a CSV file into a pandas DataFrame.

        Args:
            file_path (str): The absolute or relative path to the CSV file.

        Returns:
            Optional[pd.DataFrame]: The loaded data as a DataFrame, or None if loading failed.
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return None
