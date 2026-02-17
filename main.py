import os
import sys
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# Add the project root to the python path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from agents.reasoning_generator import ReasoningGenerator
from agents.statistical_critic import StatisticalCritic
from agents.privacy_guard import PrivacyGuard

# Define the state of our graph
class AgentState(TypedDict):
    raw_data: Optional[Any]
    reasoning_log: Optional[str]
    model_selection: Optional[str]
    privacy_status: Optional[str]

# Initialize Agents
reasoning_generator = ReasoningGenerator()
statistical_critic = StatisticalCritic()
privacy_guard = PrivacyGuard()

# Node Functions
def reasoning_node(state: AgentState) -> AgentState:
    return reasoning_generator.process(state)

def critic_node(state: AgentState) -> AgentState:
    return statistical_critic.process(state)

def privacy_node(state: AgentState) -> AgentState:
    return privacy_guard.process(state)

def main():
    print("Initializing AgentSynth-Trinity Orchestrator...")
    
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

    # Add edges
    workflow.add_edge("reasoning", "critic")
    workflow.add_edge("critic", "privacy")
    workflow.add_edge("privacy", END)

    # Set entry point
    workflow.set_entry_point("reasoning")

    # Compile the graph
    app = workflow.compile()

    # 3. Execute
    initial_state = AgentState(
        raw_data=df,
        reasoning_log=None,
        model_selection=None,
        privacy_status=None
    )
    
    print("\n--- Starting Agentic Loop ---")
    result = app.invoke(initial_state)
    print("--- Agentic Loop Completed ---\n")
    
    print("Final State:")
    print(f"Reasoning Log: {result['reasoning_log']}")
    print(f"Model Selection: {result['model_selection']}")
    print(f"Privacy Status: {result['privacy_status']}")

if __name__ == "__main__":
    main()
