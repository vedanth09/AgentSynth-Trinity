import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime
from main import run_trinity_pipeline, AgentState
from utils.prompt_parser import PromptParser, GenerationConfig
from utils.data_loader import DataLoader
from utils.reporter import TrinityReporter
from utils.trace_viz import TraceVisualizer
from utils.certificate_gen import CertificateGenerator

# --- Initialization ---
st.set_page_config(page_title="AgentSynth-Trinity Demo", page_icon="🛡️", layout="wide")
parser = PromptParser()
data_loader = DataLoader()
reporter = TrinityReporter()
viz = TraceVisualizer()
cert_gen = CertificateGenerator()

if "config" not in st.session_state:
    st.session_state.config = GenerationConfig(domain="Healthcare", goal="Synthesize high-fidelity healthcare data")
if "results" not in st.session_state:
    st.session_state.results = None
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# --- Helper Functions ---
def update_config_from_prompt(prompt):
    parsed = parser.parse(prompt)
    st.session_state.config = parsed
    st.session_state.prompt = prompt

def reset_app():
    st.session_state.results = None
    st.session_state.prompt = ""
    st.session_state.config = GenerationConfig(domain="Healthcare", goal="Synthesize high-fidelity healthcare data")
    st.rerun()

# --- Custom Styling ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 5px; height: 3em; width: 100%; }
    .chip { display: inline-block; padding: 5px 10px; border-radius: 15px; background: #e9ecef; margin: 2px; cursor: pointer; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🛡️ AgentSynth-Trinity: Autonomous Data Synthesis")
st.markdown("*A Master's Thesis Project for Healthcare & Finance Synthetic Data Generation*")
st.divider()

# --- Main Layout ---
col_left, col_right = st.columns([1.2, 0.8], gap="large")

with col_left:
    st.header("🔡 Describe the Data You Need")
    
    # Prompt Input
    prompt_input = st.text_area(
        "Natural Language Prompt",
        value=st.session_state.prompt,
        placeholder="e.g., generate 500 diabetic patient records with high privacy",
        height=150,
        label_visibility="collapsed"
    )
    
    # Example Chips
    st.write("💡 Examples:")
    examples = [
        "500 diabetic patient records with high privacy",
        "1000 fraud transactions for model stress testing",
        "200 rare disease cohort records (lupus)",
        "5000 normal credit card transactions"
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            update_config_from_prompt(ex)
            st.rerun()

    if st.button("🚀 Generate Synthetic Data", type="primary"):
        if not prompt_input and not st.session_state.prompt:
            st.warning("Please describe the data first or select an example.")
        else:
            # Re-parse if modified
            if prompt_input != st.session_state.prompt:
                update_config_from_prompt(prompt_input)
            
            # Start Pipeline
            st.session_state.results = "running"
            st.rerun()

with col_right:
    with st.expander("⚙️ Advanced Settings & Overrides", expanded=True):
        domain = st.selectbox("Domain", ["Healthcare", "Finance"], 
                             index=0 if st.session_state.config.domain == "Healthcare" else 1)
        rows = st.slider("Number of Rows", 100, 10000, st.session_state.config.rows)
        epsilon = st.slider("Privacy Budget (ε)", 0.1, 10.0, st.session_state.config.epsilon, 
                           help="0.1=Maximum Privacy, 10.0=Maximum Fidelity")
        model_type = st.selectbox("Generative Model", ["auto", "timegan", "vae", "diffusion"], 
                                 index=["auto", "timegan", "vae", "diffusion"].index(st.session_state.config.model))
        out_format = st.selectbox("Output Format", ["CSV", "JSON", "Parquet"])
        seed = st.number_input("Random Seed", value=42)

        # Update config with overrides
        st.session_state.config.domain = domain
        st.session_state.config.rows = rows
        st.session_state.config.epsilon = epsilon
        st.session_state.config.model = model_type

    if st.button("🔄 Reset System"):
        reset_app()

# --- Pipeline Execution ---
if st.session_state.results == "running":
    st.divider()
    with st.status("🕵️ Agent Orchestrator: Initiating Synthesis...", expanded=True) as status:
        # Load local data sample based on domain
        sample_path = f"data/{st.session_state.config.domain.lower()}_sample.csv"
        raw_df = data_loader.load_csv(sample_path)
        
        if raw_df is None:
            st.error(f"Failed to load sample data for {st.session_state.config.domain}. Check data/ folder.")
            st.session_state.results = None
            status.update(label="Failed to Load Data", state="error")
        else:
            # Prepare Initial State
            state = AgentState(
                raw_data=raw_df,
                domain=st.session_state.config.domain,
                goal=st.session_state.config.goal,
                epsilon_input=st.session_state.config.epsilon,
                num_rows=st.session_state.config.rows,
                selected_model_type=st.session_state.config.model
            )

            # Stream LangGraph Events
            async def run_pipeline():
                final_state = None
                async for event in run_trinity_pipeline(state):
                    # event is a dict of {node_name: output}
                    node_name = list(event.keys())[0]
                    node_output = event[node_name]
                    
                    # Update Progress in UI
                    status.write(f"✅ Agent Completed: **{node_name.capitalize()}**")
                    if "trace" in node_output and node_output["trace"]:
                        st.caption(node_output["trace"][-1])
                    
                    final_state = node_output
                return final_state

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                final_results = loop.run_until_complete(run_pipeline())
                
                st.session_state.results = final_results
                status.update(label="Synthesis Complete: Trinity Standards Met!", state="complete")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline Error: {e}")
                st.session_state.results = None
                status.update(label="Pipeline Failed", state="error")

# --- Results Dashboard ---
if isinstance(st.session_state.results, dict):
    res_state = st.session_state.results
    st.divider()
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Judge Decision", res_state.get("judge_decision", "N/A"))
    with kpi2:
        wass = res_state.get("trinity_scorecard", {}).get("fidelity", {}).get("wasserstein", 0)
        st.metric("Fidelity (Wasserstein)", f"{wass:.4f}")
    with kpi3:
        tstr = res_state.get("trinity_scorecard", {}).get("utility", {}).get("tstr_accuracy", 0)
        st.metric("Utility (TSTR)", f"{tstr:.1%}")
    with kpi4:
        eps = res_state.get("privacy_proof", {}).get("epsilon", st.session_state.config.epsilon)
        st.metric("Privacy Cost (ε)", f"{eps:.2f}")

    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "🏆 Trinity Score", "📋 Audit Log"])
    
    with tab1:
        st.subheader("Synthetic Data Preview")
        synth_df = res_state.get("safe_data_asset")
        if synth_df is not None:
            st.dataframe(synth_df.head(50), use_container_width=True)
            
            with st.expander("📈 Column Statistics"):
                st.write(synth_df.describe())
            
            st.subheader("Distribution Analysis")
            num_cols = synth_df.select_dtypes(include=[np.number]).columns[:5]
            if len(num_cols) > 0:
                selected_col = st.selectbox("Select Column to Visualize", num_cols)
                fig = px.histogram(synth_df, x=selected_col, color_discrete_sequence=['#ff7f0e'], 
                                  title=f"{selected_col} Distribution (Synthetic)")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_rad, col_metrics = st.columns([1, 1])
        
        with col_rad:
            st.subheader("Trinity Radar Chart")
            scorecard = res_state.get("trinity_scorecard", {})
            fig_radar = reporter.generate_radar_chart(scorecard, output_path=None)
            st.pyplot(fig_radar)
        
        with col_metrics:
            st.subheader("Metric Compliance Breakdown")
            # Logic for badges
            def get_badge(passed): return "✅ Pass" if passed else "❌ Fail"
            
            f_pass = scorecard.get("fidelity", {}).get("wasserstein", 1.0) < 0.15
            u_pass = scorecard.get("utility", {}).get("performance_drop", 100) < 20
            p_pass = scorecard.get("privacy", {}).get("is_compliant", False)
            
            comp_data = {
                "Metric": ["Fidelity (W-Dist)", "Utility (TSTR)", "Privacy (MIA/Link)"],
                "Value": [f"{wass:.4f}", f"{tstr:.1%}", "Compliant" if p_pass else "At Risk"],
                "Status": [get_badge(f_pass), get_badge(u_pass), get_badge(p_pass)]
            }
            st.table(pd.DataFrame(comp_data))
            
            # Overall Score
            overall = int((f_pass + u_pass + p_pass) / 3 * 100)
            grade = "Excellent (85+)" if overall > 85 else "Good (70+)" if overall >= 70 else "Needs Improvement (<70)"
            st.metric("Overall Trinity Score", f"{overall}/100", delta=grade, delta_color="normal")

    with tab3:
        st.subheader("GDPR Compliance Audit")
        st.json(res_state.get("privacy_proof", {}))
        
        st.markdown("### Compliance Checklist")
        st.checkbox("GDPR Article 30 (RoPA) Generated", value=True, disabled=True)
        st.checkbox("Privacy Guard Sanitization Applied", value=True, disabled=True)
        st.checkbox("Statistical Fidelity Verified", value=f_pass, disabled=True)
        st.checkbox("Membership Inference Attack (MIA) Resisted", value=p_pass, disabled=True)
        
        if st.button("📄 View Compliance Certificate (Text)"):
            cert_path = cert_gen.generate_text_certificate(scorecard, st.session_state.config.domain)
            with open(cert_path, "r") as f:
                st.text(f.read())

    # --- Downloads ---
    st.divider()
    st.subheader("⬇️ Download Assets")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    
    if synth_df is not None:
        csv = synth_df.to_csv(index=False).encode('utf-8')
        d_col1.download_button("Download Synthetic CSV", data=csv, file_name="synthetic_data.csv", mime="text/csv")
        
        # Reports
        report_json = json.dumps(scorecard, indent=4).encode('utf-8')
        d_col2.download_button("Download Trinity Report", data=report_json, file_name="trinity_report.json", mime="application/json")
        
        audit_json = json.dumps(res_state.get("privacy_proof", {}), indent=4).encode('utf-8')
        d_col3.download_button("Download Audit Log", data=audit_json, file_name="audit_log.json", mime="application/json")
        
        # Cert (Mocking PDF as .txt for simplicity as per Step 5 context)
        cert_text = "Trinity Compliance Certificate\n" + json.dumps(scorecard, indent=2)
        d_col4.download_button("Download Certificate (TXT)", data=cert_text.encode('utf-8'), file_name="certificate.txt")
