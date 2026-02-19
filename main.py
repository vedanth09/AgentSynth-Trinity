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
from agents.trinity_judge import TrinityJudge

# Define the state of our graph
class AgentState(TypedDict):
    # Inputs
    raw_data: Optional[Any]
    domain: str
    goal: str
    iteration: int # Track loop iterations
    epsilon_input: Optional[float] # User override for privacy budget
    
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
    
    # Judge Outputs
    trinity_scorecard: Optional[Dict[str, Any]]
    judge_decision: Optional[str]
    judge_feedback: Optional[str]

# Initialize Agents & Utils
reasoning_generator = ReasoningGenerator()
statistical_critic = StatisticalCritic()
privacy_guard = PrivacyGuard()
trinity_judge = TrinityJudge()
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
        new_state["judge_decision"] = "Reject"
        new_state["judge_feedback"] = "Privacy Guard failed to sanitize data."
    return new_state

def judge_node(state: AgentState) -> AgentState:
    return trinity_judge.evaluate(state)

def logging_node(state: AgentState) -> AgentState:
    """
    Final node to capture the audit log.
    """
    print("[Orchestrator] Logic Approved. Logging final audit trail...")
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Construct the Log
    # Merge schema skeleton with other metrics for the final log
    final_logic = state.get("schema_skeleton", {}).copy()
    final_logic["pilot_winner"] = state.get("model_selection")
    final_logic["trinity_scorecard"] = state.get("trinity_scorecard")
    
    audit_logger.log_event(
        timestamp=timestamp,
        agent_name="AgentSynth_Trinity_v1", 
        domain=state.get("domain", "Unknown"),
        goal=state.get("goal", "Unknown"),
        compliance_check=state.get("compliance_check", {}),
        logic_skeleton=final_logic,
        privacy_proof=state.get("privacy_proof", {})
    )
    return state

# Conditional Logic
def should_continue(state: AgentState):
    decision = state.get("judge_decision")
    iteration = state.get("iteration", 0)
    
    if decision == "Approve":
        return "logging"
    elif iteration >= 3:
        print("[Orchestrator] Max iterations reached. Ending loop.")
        return "logging" # Log whatever we have, even if rejected, or route to error
    else:
        print(f"[Orchestrator] Judge Rejected ({decision}). Iterating...")
        return "reasoning"

def run_trinity_pipeline(data: Any, domain: str, epsilon: Optional[float] = None) -> Dict[str, Any]:
    """
    Executes the full AgentSynth-Trinity pipeline.
    Suitable for calling from UI or external scripts.
    """
    print(f"Initializing Trinity Pipeline for {domain} (Epsilon={epsilon})...")

    # 1. Build Graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("privacy", privacy_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("logging", logging_node)

    # Add edges
    workflow.add_edge("reasoning", "critic")
    workflow.add_edge("critic", "privacy")
    workflow.add_edge("privacy", "judge")
    
    # Feedback settings
    workflow.add_conditional_edges(
        "judge",
        should_continue,
        {
            "logging": "logging",
            "reasoning": "reasoning"
        }
    )
    
    workflow.add_edge("logging", END)

    # Set entry point
    workflow.set_entry_point("reasoning")

    # Compile the graph
    app = workflow.compile()

    # 2. Execute
    initial_state = AgentState(
        raw_data=data,
        domain=domain,
        goal=f"Synthesize {domain} data corpus",
        iteration=0,
        epsilon_input=epsilon,
        reasoning_trace=None,
        compliance_check=None,
        schema_skeleton=None,
        model_selection=None,
        pilot_metrics=None,
        privacy_status=None,
        privacy_proof=None,
        safe_data_asset=None,
        trinity_scorecard=None,
        judge_decision=None,
        judge_feedback=None
    )
    
    try:
        final_state = app.invoke(initial_state)
        return final_state
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return {"judge_decision": "Error", "judge_feedback": str(e)}

def main():
    print("Initializing AgentSynth-Trinity Orchestrator (CLI Mode)...")
    
    # 1. Load Data
    data_loader = DataLoader()
    healthcare_data_path = "data/healthcare_sample.csv"
    df = data_loader.load_csv(healthcare_data_path)
    
    if df is None:
        print("Failed to load initial data. Exiting.")
        return

    # 2. Run Pipeline
    result = run_trinity_pipeline(df, "Healthcare")
    
    print("\n--- Pipeline Result ---")
    print(f"Data Shape: {result.get('safe_data_asset', pd.DataFrame()).shape}")
    print(f"Decision: {result.get('judge_decision')}")

if __name__ == "__main__":
    main()
