import streamlit as st
from typing import List, Dict, Any

class TraceVisualizer:
    """
    Utility for rendering LangGraph execution traces in Streamlit.
    """
    
    @staticmethod
    def render_trace(trace_logs: List[str]):
        """Renders the execution trace as a step-by-step timeline."""
        st.subheader("🕵️ Agent Execution Trace (Auditable Lineage)")
        
        for i, log in enumerate(trace_logs):
            with st.expander(f"Step {i+1}: {log[:50]}...", expanded=(i == len(trace_logs)-1)):
                st.info(log)
                if "Reasoning" in log:
                    st.caption("🔍 Semantic Layer Action")
                elif "critic" in log.lower():
                    st.caption("📊 Statistical Pilot Action")
                elif "privacy" in log.lower():
                    st.caption("🛡️ Privacy Guard Action")
                elif "judge" in log.lower():
                    st.caption("⚖️ Trinity Judge Action")
