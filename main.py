import os
import sys
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from datetime import datetime

# Add the project root to the python path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.audit_logger import AuditLogger
from agents.reasoning_generator import ReasoningGenerator
from agents.statistical_critic import StatisticalCritic
from agents.privacy_guard import PrivacyGuard

# Define the state of our graph
class AgentState(TypedDict):
    # Inputs
    raw_data: Optional[Any]
    domain: str
    goal: str
    
    # Reasoning Outputs
    reasoning_trace: Optional[str]
    compliance_check: Optional[Dict[str, str]]
    schema_skeleton: Optional[Dict[str, Any]]
    
    # Critic Outputs
    model_selection: Optional[str]
    pilot_metrics: Optional[Dict[str, float]]
    
    # Privacy Outputs
    privacy_status: Optional[str]
    privacy_proof: Optional[Dict[str, Any]]
    safe_data_asset: Optional[Any]

# Initialize Agents & Utils
reasoning_generator = ReasoningGenerator()
statistical_critic = StatisticalCritic()
privacy_guard = PrivacyGuard()
audit_logger = AuditLogger()

# Node Functions
def reasoning_node(state: AgentState) -> AgentState:
    return reasoning_generator.process(state)

def critic_node(state: AgentState) -> AgentState:
    return statistical_critic.process(state)

def privacy_node(state: AgentState) -> AgentState:
    # Sanitization Checkpoint
    new_state = privacy_guard.process(state)
    if new_state.get("privacy_status") != "Privacy-Cleared":
        print("!!! PRIVACY COMPLIANCE FAILED !!! - Stopping Flow")
        # In a more complex graph, we might route to an error handling node
    return new_state

def logging_node(state: AgentState) -> AgentState:
    """
    Final node to capture the audit log.
    """
    print("[Orchestrator] Logging audit trail...")
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Construct the Log
    audit_logger.log_event(
        timestamp=timestamp,
        agent_name="AgentSynth_Trinity_v1", # Unified Agent Name for final log
        domain=state.get("domain", "Unknown"),
        goal=state.get("goal", "Unknown"),
        compliance_check=state.get("compliance_check", {}),
        logic_skeleton=state.get("schema_skeleton", {
            "pilot_winner": state.get("model_selection"),
            "pilot_metrics": state.get("pilot_metrics")
        }),
        privacy_proof=state.get("privacy_proof", {})
    )
    return state

def main():
    print("Initializing AgentSynth-Trinity Orchestrator (Sprint 2)...")
    
    # 1. Load Data
    data_loader = DataLoader()
    healthcare_data_path = "data/healthcare_sample.csv"
    df = data_loader.load_csv(healthcare_data_path)
    
    if df is None:
        print("Failed to load initial data. Exiting.")
        return

    # 2. Build Graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("privacy", privacy_node)
    workflow.add_node("logging", logging_node)

    # Add edges
    workflow.add_edge("reasoning", "critic")
    workflow.add_edge("critic", "privacy")
    workflow.add_edge("privacy", "logging")
    workflow.add_edge("logging", END)

    # Set entry point
    workflow.set_entry_point("reasoning")

    # Compile the graph
    app = workflow.compile()

    # 3. Execute
    initial_state = AgentState(
        raw_data=df,
        domain="Healthcare",
        goal="Synthesize rare disease cohort (Disease_X)",
        reasoning_trace=None,
        compliance_check=None,
        schema_skeleton=None,
        model_selection=None,
        pilot_metrics=None,
        privacy_status=None
    )
    
    print("\n--- Starting Agentic Loop ---")
    try:
        result = app.invoke(initial_state)
        print("--- Agentic Loop Completed ---\n")
    except Exception as e:
        print(f"Error during execution: {e}")
        return

if __name__ == "__main__":
    main()
