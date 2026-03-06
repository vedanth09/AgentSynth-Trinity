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
    selected_model_type: str = "auto"  # auto, timegan, vae, diffusion
    output_format: str = "CSV"         # FIX: added so app.py can pass it in
    seed: int = 42                     # FIX: added so app.py can pass it in

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


# --- FIX: Agent instances are created per-request (factory functions) to avoid
#     shared mutable state across concurrent Streamlit sessions. ---
def _make_agents():
    return (
        ReasoningGenerator(),
        StatisticalCritic(),
        PrivacyGuard(),
        TrinityJudge(),
        MetaAgent(),
        AuditLogger(),
        TrinityReporter(),
    )


# --- Async Node Functions ---
# FIX: Nodes now receive a plain dict (as LangGraph passes after serialisation)
#      and return a plain dict so the shape is consistent across all nodes.

async def meta_agent_node(state) -> dict:
    state = _to_dict(state)
    meta_agent = _make_agents()[4]
    state["trace"] = state.get("trace", []) + ["Meta-Agent: Consulting historical knowledge base..."]
    return meta_agent.process(state)

async def reasoning_node(state) -> dict:
    state = _to_dict(state)
    reasoning_generator = _make_agents()[0]
    state["trace"] = state.get("trace", []) + ["Reasoning Generator: Analyzing domain semantics..."]
    return await reasoning_generator.process(state)

async def critic_node(state) -> dict:
    state = _to_dict(state)
    statistical_critic = _make_agents()[1]
    state["trace"] = state.get("trace", []) + ["Statistical Critic: Running experimental pilot and ensembling..."]
    return await statistical_critic.process(state)

async def privacy_node(state) -> dict:
    state = _to_dict(state)
    privacy_guard = _make_agents()[2]
    state["trace"] = state.get("trace", []) + ["Privacy Guard: Applying (ε, δ)-DP constraints with RDP-accounting..."]
    return await privacy_guard.process(state)

async def judge_node(state) -> dict:
    state = _to_dict(state)
    trinity_judge = _make_agents()[3]
    state["trace"] = state.get("trace", []) + ["Trinity Judge: Commencing multi-dimensional evaluation (Fidelity, Utility, Privacy)..."]
    return await trinity_judge.evaluate(state)

async def logging_node(state) -> dict:
    state = _to_dict(state)
    audit_logger = _make_agents()[5]
    state["trace"] = state.get("trace", []) + ["Orchestrator: Flow Approved. Finalizing audit lineage."]
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    audit_logger.log_event(
        timestamp=timestamp,
        agent_name="Trinity_v2_Pydantic",
        domain=state.get("domain"),
        goal=state.get("goal"),
        compliance_check=state.get("compliance_check"),
        logic_skeleton=state.get("schema_skeleton"),
        privacy_proof=state.get("privacy_proof"),
    )
    return state


# --- Conditional routing ---
# FIX: `should_continue` now accepts a plain dict (what LangGraph actually passes
#      after node execution) instead of an AgentState instance.
def should_continue(state) -> str:
    state = _to_dict(state)
    judge_decision = state.get("judge_decision")
    iteration = state.get("iteration", 0)
    judge_feedback = state.get("judge_feedback", "") or ""

    if judge_decision == "Approve" or iteration >= 3:
        return "logging"
    return "critic" if "Fidelity" in judge_feedback else "reasoning"


# --- Helper: normalise whatever LangGraph passes into a plain dict ---
def _to_dict(state) -> dict:
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return dict(state)


# --- Graph Assembly ---
def build_graph():
    # Use dict as the state schema so LangGraph always passes plain dicts to
    # nodes — avoids the AgentState.get() AttributeError seen with Pydantic schemas.
    workflow = StateGraph(dict)
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

    workflow.add_conditional_edges(
        "judge",
        should_continue,
        {"critic": "critic", "reasoning": "reasoning", "logging": "logging"},
    )
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