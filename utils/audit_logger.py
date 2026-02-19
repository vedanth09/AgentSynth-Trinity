import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

class AuditLogger:
    """
    Handles logging of agent reasoning, compliance checks, and logic skeletons
    in a structured JSON format.
    """

    def __init__(self, log_file: str = "audit_log.json"):
        self.log_file = log_file
        # Configure logging to console as well
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger("AuditLogger")

    def log_event(self, 
                  timestamp: str,
                  agent_name: str,
                  domain: str,
                  goal: str,
                  compliance_check: Dict[str, str],
                  logic_skeleton: Dict[str, Any],
                  privacy_proof: Optional[Dict[str, Any]] = None):
        """
        Logs a structured event to the JSON log file.

        Args:
            timestamp (str): ISO 8601 timestamp.
            agent_name (str): Name of the agent generating the log.
            domain (str): Domain of operation (e.g., "Healthcare").
            goal (str): The specific business goal.
            compliance_check (Dict[str, str]): GDPR framework and reasoning.
            logic_skeleton (Dict[str, Any]): Constraints and fidelity targets.
            privacy_proof (Optional[Dict[str, Any]]): Epsilon and noise details.
        """
        
        log_entry = {
            "timestamp": timestamp,
            "agent": agent_name,
            "domain": domain,
            "goal": goal,
            "compliance_check": compliance_check,
            "logic_skeleton": logic_skeleton,
            "privacy_proof": privacy_proof
        }

        # Print to console for immediate visibility
        self.logger.info(f"\n[AUDIT LOG] New Entry:\n{json.dumps(log_entry, indent=2)}")

        # Append to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write to audit log: {e}")
            
    def generate_current_timestamp(self) -> str:
        """Returns current UTC timestamp in ISO format."""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
