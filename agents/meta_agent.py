import chromadb
from typing import Dict, Any, List, Optional
import json

class MetaAgent:
    """
    Meta-Agent layer for dynamic model selection based on historical performance.
    
    Academic Motivation: Meta-learning for model selection helps automate the choice 
    of inductive biases by leveraging empirical performance on similar datasets 
    (Vanschoren, 2018). In AgentSynth-Trinity, this minimizes redundant piloting 
    of sub-optimal generators by retrieving the best-performing models for similar 
    domain corpora and modalities.
    """
    
    def __init__(self, db_path: str = "./data/meta_store"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="model_performance")
        self._seed_initial_knowledge()

    def _seed_initial_knowledge(self):
        """Seeds the vector DB with initial research-based model strengths."""
        if self.collection.count() == 0:
            # Baseline knowledge: TVAE is good for tabular, TimeGAN for time-series.
            self.collection.add(
                ids=["healthcare_tabular", "finance_timeseries", "general_tabular"],
                metadatas=[
                    {"best_model": "TVAE", "modality": "tabular", "domain": "Healthcare"},
                    {"best_model": "TimeGAN", "modality": "time-series", "domain": "Finance"},
                    {"best_model": "GaussianCopula", "modality": "tabular", "domain": "General"}
                ],
                documents=[
                    "High-dimensional healthcare records with mixed types.",
                    "Sequential financial transaction data with temporal correlations.",
                    "General-purpose low-dimensional tabular data."
                ]
            )

    def suggest_model(self, domain: str, modality: str, description: Optional[str] = None) -> str:
        """
        Retrieves the recommended model from historical performance data.
        """
        query_text = f"{domain} {modality} {description or ''}"
        results = self.collection.query(
            query_texts=[query_text],
            n_results=1
        )
        
        if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
            recommendation = results['metadatas'][0][0]['best_model']
            print(f"[MetaAgent] Recommended Model: {recommendation} (based on history)")
            return recommendation
        
        return "GaussianCopula" # Default fallback

    def update_history(self, domain: str, modality: str, model_name: str, score: float, description: str):
        """
        Stores new performance results in the vector DB for future reference.
        """
        self.collection.add(
            ids=[f"{domain}_{modality}_{model_name}_{score:.4f}"],
            metadatas={"best_model": model_name, "score": score, "domain": domain, "modality": modality},
            documents=[description]
        )

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent node for Suggeting models.
        """
        domain = state.get("domain", "General")
        modality = "tabular" # Default for now, could be inferred from data
        description = state.get("goal", "Data synthesis")
        
        recommended = self.suggest_model(domain, modality, description)
        state["recommended_model"] = recommended
        
        return state
