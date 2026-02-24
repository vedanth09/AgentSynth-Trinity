import os
import sys
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ConfigDict
from langgraph.graph import StateGraph, END
from datetime import datetime

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.audit_logger import AuditLogger
from agents.reasoning_generator import ReasoningGenerator
from agents.statistical_critic import StatisticalCritic
from agents.privacy_guard import PrivacyGuard
from agents.trinity_judge import TrinityJudge
from agents.meta_agent import MetaAgent
from utils.reporter import TrinityReporter

# --- Agent State Definition (Pydantic v2) ---
class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Inputs
    raw_data: Optional[Any] = None
    domain: str = "Healthcare"
    goal: str = "General Synthesis"
    iteration: int = 0
    epsilon_input: Optional[float] = 1.0
    num_rows: Optional[int] = 100
    selected_model_type: str = "auto" # auto, timegan, vae, diffusion
    
    # Outputs
    recommended_model: Optional[str] = None
    reasoning_trace: Optional[str] = None
    compliance_check: Optional[Dict[str, Any]] = None
    schema_skeleton: Optional[Dict[str, Any]] = None
    model_selection: Optional[str] = None
    pilot_metrics: Optional[Dict[str, float]] = None
    privacy_status: Optional[str] = None
    privacy_proof: Optional[Dict[str, Any]] = None
    safe_data_asset: Optional[Any] = None
    trinity_scorecard: Optional[Dict[str, Any]] = None
    judge_decision: Optional[str] = None
    judge_feedback: Optional[str] = None
    trace: List[str] = []

# --- Initialize Global Components ---
reasoning_generator = ReasoningGenerator()
statistical_critic = StatisticalCritic()
privacy_guard = PrivacyGuard()
trinity_judge = TrinityJudge()
meta_agent = MetaAgent()
audit_logger = AuditLogger()
reporter = TrinityReporter()

# --- Async Node Functions ---
async def meta_agent_node(state: AgentState) -> Dict:
    state.trace.append("Meta-Agent: Consulting historical knowledge base...")
    return meta_agent.process(state.model_dump())

async def reasoning_node(state: AgentState) -> Dict:
    state.trace.append("Reasoning Generator: Analyzing domain semantics...")
    return await reasoning_generator.process(state.model_dump())

async def critic_node(state: AgentState) -> Dict:
    state.trace.append("Statistical Critic: Running experimental pilot and ensembling...")
    return await statistical_critic.process(state.model_dump())

async def privacy_node(state: AgentState) -> Dict:
    state.trace.append("Privacy Guard: Applying (ε, δ)-DP constraints with RDP-accounting...")
    return await privacy_guard.process(state.model_dump())

async def judge_node(state: AgentState) -> Dict:
    state.trace.append("Trinity Judge: Commencing multi-dimensional evaluation (Fidelity, Utility, Privacy)...")
    return await trinity_judge.evaluate(state.model_dump())

async def logging_node(state: AgentState) -> Dict:
    state.trace.append("Orchestrator: Flow Approved. Finalizing audit lineage.")
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    audit_logger.log_event(
        timestamp=timestamp, agent_name="Trinity_v2_Pydantic", 
        domain=state.domain, goal=state.goal,
        compliance_check=state.compliance_check,
        logic_skeleton=state.schema_skeleton,
        privacy_proof=state.privacy_proof
    )
    return state.model_dump()

# --- Graph Assembly ---
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("meta_agent", meta_agent_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("privacy", privacy_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("logging", logging_node)
    
    workflow.set_entry_point("meta_agent")
    workflow.add_edge("meta_agent", "reasoning")
    workflow.add_edge("reasoning", "critic")
    workflow.add_edge("critic", "privacy")
    workflow.add_edge("privacy", "judge")
    
    def should_continue(state: AgentState):
        if state.judge_decision == "Approve" or state.iteration >= 3:
            return "logging"
        return "critic" if "Fidelity" in (state.judge_feedback or "") else "reasoning"

    workflow.add_conditional_edges("judge", should_continue, {"critic": "critic", "reasoning": "reasoning", "logging": "logging"})
    workflow.add_edge("logging", END)
    return workflow.compile()

async def run_trinity_pipeline(initial_state: AgentState):
    """
    Exposes the Trinity pipeline as an async generator for real-time streaming.
    """
    app = build_graph()
    async for event in app.astream(initial_state.model_dump()):
        yield event

if __name__ == "__main__":
    # CLI fallback for testing
    import pandas as pd
    data = pd.DataFrame({"age": [25, 30], "income": [50000, 60000]})
    config = AgentState(raw_data=data, domain="Healthcare")
    async def test_run():
        async for step in run_trinity_pipeline(config):
            print(f"Step: {list(step.keys())[0]}")
    asyncio.run(test_run())
