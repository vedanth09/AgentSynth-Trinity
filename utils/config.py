class PrivacyBudget:
    """
    Configuration for Differential Privacy Budgets (Epsilon).
    Lower Epsilon = Stronger Privacy, Lower Utility.
    Higher Epsilon = Weaker Privacy, Higher Utility.
    """
    LOW = 0.1     # Very strict, high noise
    MEDIUM = 1.0  # Balanced
    HIGH = 10.0   # Loose, low noise
    
    @staticmethod
    def get_budget(risk_level: str) -> float:
        """
        Returns the epsilon value for a given risk level.
        """
        mapping = {
            "critical": PrivacyBudget.LOW,
            "high": PrivacyBudget.LOW,
            "medium": PrivacyBudget.MEDIUM,
            "low": PrivacyBudget.HIGH,
            "public": PrivacyBudget.HIGH
        }
        return mapping.get(risk_level.lower(), PrivacyBudget.MEDIUM)
