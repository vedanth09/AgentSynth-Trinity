# AgentSynth-Trinity Architecture & Data Flow

This document provides the high-level architecture and data flow pipeline for the AgentSynth-Trinity research framework.

## 🛡️ System Architecture

![AgentSynth-Trinity Architecture Visualization](/Users/vedanth/.gemini/antigravity/brain/f6aba82b-4276-4e11-aaaa-eafd08897b6b/agentsynth_trinity_architecture_viz_1771095137419.png)

The following diagram illustrates the multi-agent orchestrator and the "Trinity" evaluation framework.

```mermaid
graph TD
    subgraph Input_Layer ["Input & Knowledge Layer"]
        RD[(Raw Data)]
        DK[Domain Knowledge / Guidelines]
    end

    subgraph Agentic_Pipeline ["Multi-Agent Orchestrator (LangGraph/NeMo)"]
        RG["Reasoning Generator (LLM)"]
        SC["Statistical Critic (GAN/VAE/Diffusion)"]
        PG["Privacy Guard (Opacus/SmartNoise)"]
    end

    subgraph Trinity_Framework ["💎 Trinity Evaluation Framework"]
        F["Fidelity (Wasserstein Distance)"]
        U["Utility (ML Performance)"]
        P["Privacy (DP/Re-id Risk)"]
    end

    RD --> RG
    DK --> RG
    RG --> SC
    SC --> PG
    PG --> Trinity_Framework
    
    Trinity_Framework -- "Feedback Loop" --> RG
    Trinity_Framework -- "Validation Passed" --> Output[(Compliant Synthetic Data)]
```

---

## 💹 Data Flow Pipeline

Detailed flow of data from ingestion to compliant output.

```mermaid
sequenceDiagram
    participant User as Data Scientist
    participant RG as Reasoning Generator
    participant SC as Statistical Critic
    participant PG as Privacy Guard
    participant Trinity as Trinity Eval Engine

    User->>RG: Upload Raw Data & Domain Specs
    Note right of RG: Chain-of-Thought Logic applied
    RG->>SC: Draft Synthetic Candidates
    Note right of SC: Selects Optimal Model (TimeGAN/VAE)
    SC->>PG: Optimized Synthetic Data
    Note right of PG: Controlled Noise Injection (DP)
    PG->>Trinity: Private Synthetic Dataset
    
    loop Evaluation Loop
        Trinity->>Trinity: Fidelity check (Correlation/Wasserstein)
        Trinity->>Trinity: Utility check (XGBoost/Random Forest)
        Trinity->>Trinity: Privacy check (DP Audit)
        alt Fails Benchmarks
            Trinity-->>RG: Feedback (Improve Logic/Add Noise)
        else Passes Benchmarks
            Trinity->>User: Validated Synthetic Dataset + Audit Logs
        end
    end
```
