import pandas as pd
from typing import Dict, List, Any

class DomainIntelligence:
    """
    Utilities for semantic validation against industry standards.
    
    Academic Motivation: Grounding LLM reasoning in industry ontologies 
    (FHIR, Basel III) ensures that synthesized data is semantically viable 
    and regulatory-compliant.
    """
    
    FHIR_REQUIRED_FIELDS = ["patient_id", "timestamp", "observation_code", "value"]
    BASEL_III_RISK_FIELDS = ["exposure_at_default", "probability_of_default", "loss_given_default"]

    @staticmethod
    def check_fhir_alignment(columns: List[str]) -> Dict[str, Any]:
        """Checks if the schema aligns with basic FHIR Observation resources."""
        missing = [f for f in DomainIntelligence.FHIR_REQUIRED_FIELDS if f not in columns]
        is_aligned = len(missing) == 0
        return {
            "is_aligned": is_aligned,
            "missing_fields": missing,
            "standard": "FHIR R4 (Observation Resource)"
        }

    @staticmethod
    def check_basel_compliance(columns: List[str]) -> Dict[str, Any]:
        """Checks for presence of Basel III credit risk parameters."""
        found = [f for f in DomainIntelligence.BASEL_III_RISK_FIELDS if f in columns]
        is_compliant = len(found) > 0
        return {
            "is_compliant": is_compliant,
            "found_risk_parameters": found,
            "standard": "Basel III (Credit Risk Framework)"
        }
