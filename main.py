import os
import sys
import asyncio
import streamlit as st
import pandas as pd
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
from utils.trace_viz import TraceVisualizer

# --- Agent State Definition (Pydantic v2) ---
class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Inputs
    raw_data: Optional[Any] = None
    domain: str = "Healthcare"
    goal: str = "General Synthesis"
    iteration: int = 0
    epsilon_input: Optional[float] = 1.0
    
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
viz = TraceVisualizer()

# --- Async Node Functions ---
async def meta_agent_node(state: AgentState) -> Dict:
    state.trace.append("Meta-Agent: Consulting historical knowledge base...")
    # LangGraph expectation: return a dictionary overlay
    res = meta_agent.process(state.model_dump())
    return res

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
    state.trace.append("Orchestrator: Flow Approved. Finalizing audit lineage and compliance records.")
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

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="AgentSynth-Trinity v2", page_icon="🛡️", layout="wide")
    
    st.title("🛡️ AgentSynth-Trinity: Research-Grade Data Synthesis")
    st.markdown("### Multi-Agent Orchestration for Privacy-Preserving GenAI (v2.0)")

    st.sidebar.header("🎛️ Synthesis Parameters")
    domain = st.sidebar.selectbox("Domain", ["Healthcare", "Finance"])
    epsilon = st.sidebar.slider("Privacy Budget (ε)", 0.1, 5.0, 1.0)
    st.sidebar.divider()
    st.sidebar.markdown("#### Tech Stack")
    st.sidebar.code("LangGraph | ChromaDB | MLflow | Pydantic v2")

    uploaded_file = st.file_uploader("Upload Real Dataset (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} records.")
        
        if st.button("🚀 Execute Trinity Pipeline"):
            with st.status("Orchestrating Agents...", expanded=True) as status:
                initial_state = AgentState(
                    raw_data=df, domain=domain, iteration=0, epsilon_input=epsilon,
                    goal=f"Synthesize {domain} dataset via AgentSynth-Trinity"
                )
                
                app = build_graph()
                final_output = asyncio.run(app.ainvoke(initial_state))
                # State in LangGraph is updated dict-wise; cast back for easier UI access
                final_state = AgentState(**final_output)
                status.update(label="Synthesis Complete!", state="complete")

            # --- Results ---
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("⚖️ Trinity Scorecard (Radar Visualization)")
                if final_state.trinity_scorecard:
                    fig = reporter.generate_radar_chart(final_state.trinity_scorecard)
                    st.pyplot(fig)
                
                st.subheader("📊 Quantitative Benchmark Metrics")
                st.json(final_state.trinity_scorecard)

            with col2:
                st.subheader("📜 Compliance Registry")
                if final_state.judge_decision == "Approve":
                    st.success(f"Decision: **COMPLIANCE APPROVED**")
                else:
                    st.error(f"Decision: **{final_state.judge_decision}**")
                
                st.write(f"Refinement Trace: {final_state.judge_feedback}")
                
                if final_state.safe_data_asset is not None:
                    csv = final_state.safe_data_asset.to_csv(index=False)
                    if st.download_button("📥 Download Synthetic Data", csv, "synthetic_data.csv", "text/csv"):
                        st.balloons()

            # --- Audit & Trace ---
            st.divider()
            viz.render_trace(final_state.trace)
            st.subheader("📁 Record of Processing Activity (RoPA)")
            st.json(final_state.privacy_proof or {})

if __name__ == "__main__":
    main()
