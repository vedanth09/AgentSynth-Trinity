from typing import Dict, Any, List
import json
import random

class ReasoningGenerator:
    """
    Agent responsible for generating semantic logic and constraints 
    based on domain requirements, ensuring privacy compliance.
    """
    
    def __init__(self):
        self.prompts = {
            "Healthcare": """
            Role: Expert Medical Data Scientist & Privacy Officer.
            Task: Define constraints for a synthetic healthcare dataset.
            Focus: Clinical plausibility (e.g., correlating age, diagnosis, lab results).
            Regulation: STRICTLY adhere to GDPR Recital 26.
            """,
            "Finance": """
            Role: Expert Financial Analyst & Compliance Officer.
            Task: Define constraints for a synthetic financial transaction dataset.
            Focus: Temporal transaction logic (e.g., sequence of fraud events, balance consistency).
            Regulation: STRICTLY adhere to GDPR Recital 26.
            """
        }

    def _generate_cot_reasoning(self, domain: str, goal: str) -> Dict[str, Any]:
        """
        Simulates the LLM's Chain-of-Thought generation process.
        In a real system, this would call an LLM API.
        """
        
        # Simulated CoT artifacts based on domain
        if domain == "Healthcare":
            compliance_reasoning = (
                "Determined that standard pseudonymization is insufficient for "
                f"the goal '{goal}' due to re-identification risk of rare attributes. "
                "Recital 26 requires that data be rendered anonymous in such a way "
                "that the data subject is no longer identifiable. "
                "Instructing Privacy Guard to apply a lower epsilon budget (ε=0.5) "
                "to ensure high anonymity while maintaining the statistical correlation "
                "between Age and Diagnosis."
            )
            constraints = [
                "Age > 18",
                "Diagnosis: Correlated with Age (e.g., Type 2 Diabetes more common > 40)",
                "Treatment_Cost: Log-normally distributed",
                "Lab_Values: Consistent with Diagnosis"
            ]
            fidelity_target = "Wasserstein Distance < 0.1 for numerical columns"
            
        elif domain == "Finance":
            compliance_reasoning = (
                "Assessing transaction data generation under GDPR Recital 26. "
                "Financial patterns can be highly fingerprintable. "
                "To prevent singularity attacks, we must mask exact timestamps and "
                "aggregate rare high-value transactions. "
                "Directing Privacy Guard to use differential privacy on 'Amount' "
                "and 'Location' fields."
            )
            constraints = [
                "Transaction_Sequence: Chronological order per User_ID",
                "Balance: Cannot be negative after transaction",
                "Fraud_Flag: < 1% of total transactions",
                "Category: Consistent with Merchant_ID"
            ]
            fidelity_target = "Autocorrelation error < 0.05"
        else:
             compliance_reasoning = "Standard GDPR compliance check."
             constraints = ["General data consistency"]
             fidelity_target = "Statistical similarity"

        return {
            "compliance_check": {
                "framework": "GDPR Recital 26",
                "reasoning": compliance_reasoning
            },
            "logic_skeleton": {
                "constraints": constraints,
                "fidelity_target": fidelity_target
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input state to generate semantic logic and audit logs.
        Handles feedback loops from Trinity Judge.
        """
        domain = state.get("domain", "Healthcare")
        goal = state.get("goal", "General Synthesis")
        iteration = state.get("iteration", 0)
        feedback = state.get("judge_feedback", None)
        
        print(f"[ReasoningGenerator] Generating logic (Iter {iteration}). Domain: {domain}")
        
        # 1. Select Prompt
        system_prompt = self.prompts.get(domain, self.prompts["Healthcare"])
        
        # 2. Refine Logic based on Feedback
        if feedback and iteration > 0:
            print(f"   > Refining based on Feedback: {feedback}")
            # Simulated refining logic
            if "Increase Noise" in feedback:
                 reasoning_note = "Received privacy alert. Increasing Epsilon strictness."
                 # Logic to trigger lower epsilon in Privacy Guard via reasoning text
            elif "Fidelity too low" in feedback:
                 reasoning_note = "Fidelity issue detected. Enforcing stricter constraints on correlation."
            else:
                 reasoning_note = f"Adjusting generation parameters based on: {feedback}"
        else:
            reasoning_note = "Initial generation pass."

        # 3. Generate CoT (Simulated)
        reasoning_output = self._generate_cot_reasoning(domain, goal)
        
        # Append refinement note to compliance reasoning for observability
        original_reasoning = reasoning_output["compliance_check"]["reasoning"]
        reasoning_output["compliance_check"]["reasoning"] = f"{original_reasoning} [Refinement: {reasoning_note}]"
        
        # 4. Update State
        state["compliance_check"] = reasoning_output["compliance_check"]
        state["schema_skeleton"] = reasoning_output["logic_skeleton"]
        state["reasoning_trace"] = f"Processed with prompt: {system_prompt.strip()} | Refinement: {reasoning_note}"
        
        return state
