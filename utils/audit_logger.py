import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

class AuditLogger:
    """
    Handles logging of agent reasoning, compliance checks, and synthesis lineage
    in a structured format compliant with GDPR Article 30 (RoPA).
    
    Academic Motivation: GDPR Article 30 mandates that controllers maintain 
    a record of processing activities (RoPA). This includes documenting the 
    technical and organizational measures (TOMs) like Differential Privacy.
    """

    def __init__(self, log_file: str = "audit_log.json"):
        self.log_file = log_file
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger("AuditLogger")

    def log_event(self, 
                  timestamp: str,
                  agent_name: str,
                  domain: str,
                  goal: str,
                  compliance_check: Dict[str, Any],
                  logic_skeleton: Dict[str, Any],
                  privacy_proof: Optional[Dict[str, Any]] = None):
        """
        Logs a structured event according to GDPR Article 30 (RoPA) standards.
        """
        
        # RoPA Structure (Article 30 Compliance)
        ropa_entry = {
            "ropa_metadata": {
                "controller": "AgentSynth-Trinity System",
                "purpose_of_processing": goal,
                "data_categories": "Sensitive " + domain + " Tabular Data",
                "storage_limitation": "Ephemeral (Session-based)",
                "technical_measures": "Differential Privacy (ε, δ)-DP",
                "lawful_basis": "Art. 6(1)(f) - Legitimate Interests"
            },
            "technical_lineage": {
                "timestamp": timestamp,
                "agent": agent_name,
                "domain": domain,
                "reasoning_trace": compliance_check.get("reasoning", "N/A"),
                "synthesis_logic": logic_skeleton,
                "privacy_parameters": privacy_proof
            }
        }

        self.logger.info(f"\n[GDPR ART. 30 AUDIT] New Record of Processing Activity (RoPA) logged for {domain}.")

        # Append to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(ropa_entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write to RoPA audit log: {e}")
            
    def generate_current_timestamp(self) -> str:
        """Returns current UTC timestamp in ISO format."""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
