import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from main import run_trinity_pipeline
from utils.data_loader import DataLoader

# --- Page Config & Styling ---
st.set_page_config(
    page_title="AgentSynth-Trinity", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Research-Grade" Aesthetics
st.markdown("""
<style>
    /* Main Background & Text */
    .reportview-container {
        background: #f0f2f6;
    }
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #0e1117;
        font-weight: 700;
    }
    h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #262730;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #0068c9;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        color: #0068c9;
        font-weight: 600;
    }
    
    /* Success/Error Banners */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# 🛡️") 
with col_title:
    st.title("AgentSynth-Trinity")
    st.markdown("**Autonomous Generative Intelligence for Privacy-Preserving Synthetic Data**")
    st.markdown("`Fidelity` • `Utility` • `Privacy`")

st.markdown("---")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    st.subheader("1. Data Source")
    domain = st.selectbox("Select Domain Corpus", ["Healthcare", "Finance"], index=0)
    
    st.subheader("2. Privacy Budget")
    epsilon = st.slider(
        "Epsilon (ε)", 
        min_value=0.1, 
        max_value=10.0, 
        value=1.0, 
        step=0.1,
        help="Controls the privacy-utility trade-off. Lower ε = Stronger Privacy."
    )
    
    # Visual indicator for privacy level
    if epsilon < 1.0:
        st.caption("🔒 **Strict Privacy Mode** (High Noise)")
    elif epsilon < 5.0:
        st.caption("🛡️ **Balanced Mode** (Moderate Noise)")
    else:
        st.caption("🚀 **High Utility Mode** (Low Noise)")

    st.subheader("3. Transparency")
    show_reasoning = st.checkbox("Show Agent Reasoning Chains", value=True)
    
    st.markdown("---")
    
    generate_btn = st.button("🚀 Initiating Synthesis", type="primary", use_container_width=True)

    st.markdown("### System Status")
    st.info("Orchestrator: **Ready**")

# --- Main Logic ---
if generate_btn:
    with st.spinner("🤖 **Agents at work...** (Reasoning → Critiquing → Sanitizing → Judging)"):
        # Load correct dataset based on domain
        data_loader = DataLoader()
        file_path = "data/healthcare_sample.csv" if domain == "Healthcare" else "data/finance_sample.csv"
        raw_data = data_loader.load_csv(file_path)
        
        if raw_data is not None:
            # Run the Pipeline
            result_state = run_trinity_pipeline(raw_data, domain, epsilon)
            
            # Store inputs/outputs in session state for persistence
            st.session_state["raw_data"] = raw_data
            st.session_state["synthetic_data"] = result_state.get("safe_data_asset")
            st.session_state["result_state"] = result_state
            
            # st.toast("Synthesis Complete!", icon="✅")
        else:
            st.error("Failed to load source data.")

# --- Dashboard Display ---
if "result_state" in st.session_state:
    state = st.session_state["result_state"]
    real = st.session_state["raw_data"]
    synth = st.session_state["synthetic_data"]
    
    # Trinity Scorecard & Decision Banner
    scorecard = state.get("trinity_scorecard", {})
    decision = state.get("judge_decision", "Unknown")
    feedback = state.get("judge_feedback", "")
    iteration = state.get("iteration", 0)

    # Top-level KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("Judge Decision", decision, delta="Approved" if decision == "Approve" else "-Rejected", delta_color="normal")
    with kpi2:
        wass = scorecard.get('fidelity', {}).get('wasserstein', 0)
        st.metric("Fidelity (Wasserstein)", f"{wass:.3f}", delta=f"{0.1-wass:.3f} margin", delta_color="inverse")
    with kpi3:
        acc = scorecard.get('utility', {}).get('tstr_accuracy', 0)
        st.metric("Utility (TSTR Acc)", f"{acc:.1%}", delta="Model Quality")
    with kpi4:
        proof = state.get("privacy_proof", {})
        used_eps = proof.get('epsilon', 0)
        st.metric("Privacy Cost (ε)", f"{used_eps:.1f}", delta="Budget Consumed", delta_color="off")

    if decision != "Approve":
         st.error(f"**Action Required**: System rejected data (Iter {iteration}). Feedback: {feedback}")


    # Tabs Layout
    st.markdown("### Deep Dive Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Fidelity & Statistics", "⚙️ Downstream Utility", "🔒 Privacy Audit", "🧠 Agent Reasoning"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Distribution Overlap")
            numeric_cols = real.select_dtypes(include=[np.number]).columns
            selected_col = st.selectbox("Compare Column Distribution", numeric_cols)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=real[selected_col], name='Real Data', opacity=0.6, marker_color='#1f77b4'))
            fig.add_trace(go.Histogram(x=synth[selected_col], name='Synthetic Data', opacity=0.6, marker_color='#ff7f0e'))
            fig.update_layout(
                barmode='overlay', 
                title_text=f"{selected_col} Distribution",
                xaxis_title=selected_col,
                yaxis_title="Count",
                template="plotly_white",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Correlation Matrix Difference")
            # Calculate difference for visualization
            real_corr = real[numeric_cols].corr()
            synth_corr = synth[numeric_cols].corr()
            corr_diff = real_corr - synth_corr
            
            # Plot heatmap using Plotly for interactivity
            fig_corr = px.imshow(
                corr_diff, 
                text_auto=True, 
                color_continuous_scale='RdBu_r', 
                range_color=[-0.5, 0.5],
                title="Correlation Error (Real - Synthetic)"
            )
            fig_corr.update_layout(template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("*Values close to 0 mean high fidelity preservation of relationships.*")

    with tab2:
        st.markdown("#### Machine Learning Efficacy (TSTR)")
        util_scores = scorecard.get("utility", {})
        
        # Comparison Bar Chart
        models = ["Real Data (Baseline)", "Synthetic Data (TSTR)"]
        accuracies = [util_scores.get('baseline_accuracy', 0), util_scores.get('tstr_accuracy', 0)]
        
        fig_util = go.Figure(data=[
            go.Bar(name='Accuracy', x=models, y=accuracies, marker_color=['#2ca02c', '#17becf'])
        ])
        fig_util.update_layout(
            title="XGBoost Classifier Performance",
            yaxis_title="Accuracy",
            template="plotly_white",
            yaxis_tickformat='.0%'
        )
        st.plotly_chart(fig_util, use_container_width=True)
        
        st.info(f"**Insight**: The synthetic data retained **{100 - util_scores.get('performance_drop', 0):.1f}%** of the original data's predictive power.")

    with tab3:
        st.markdown("#### Differential Privacy Verification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            proof = state.get("privacy_proof", {})
            st.markdown(f"**Mechanism**: `{proof.get('mechanism', 'Unknown')}`")
            st.markdown(f"**Noise Distribution**: `{proof.get('noise_distribution', 'Unknown')}`")
            
            # Mock Gauge Chart for Risk
            eps_val = proof.get('epsilon', 0)
            max_val = 10.0
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = eps_val,
                title = {'text': "Privacy Loss (ε)"},
                gauge = {
                    'axis': {'range': [None, max_val]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgreen"},
                        {'range': [1, 5], 'color': "yellow"},
                        {'range': [5, 10], 'color': "orange"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 9.9}
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)


        with col2:
            st.success(f"✅ **GDPR Compliance**: {proof.get('compliance_statement', 'Pending')}")
            
            privacy_card = scorecard.get("privacy", {})
            matches = privacy_card.get("exact_matches", 0)
            if matches == 0:
                st.success(f"**Linkability Check Passed**: 0 Exact Row Matches found.")
            else:
                st.error(f"**Linkability Check FAILED**: {matches} Exact Matches found!")

    with tab4:
        if show_reasoning:
            st.markdown("#### 🧠 Autonomous Agent Reasoning")
            
            with st.expander("Step 1: Reasoning Generator (Chain-of-Thought)", expanded=True):
                 st.markdown(f"**Goal**: {state.get('goal')}")
                 st.text_area("Reasoning Trace", state.get("reasoning_trace"), height=150, key="trace_1")
                 st.json(state.get("compliance_check"))
            
            with st.expander("Step 2: Statistical Critic (Model Selection)"):
                 st.markdown(f"**Selected Model**: `{state.get('model_selection')}`")
                 st.json(state.get("pilot_metrics"))
                 
            with st.expander("Step 3: Privacy Guard (Sanitization)"):
                 st.write(state.get("privacy_proof"))

            if state.get("judge_feedback"):
                st.error(f"⚠️ **Feedback Loop Triggered**: {state.get('judge_feedback')}")

else:
    # Empty State Hero
    st.markdown(
        """
        <div style="text-align: center; padding: 50px;">
            <h3>👋 Welcome to AgentSynth-Trinity for Streamlit</h3>
            <p>Select your parameters in the sidebar and click <b>Initiate Synthesis</b> to begin.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
