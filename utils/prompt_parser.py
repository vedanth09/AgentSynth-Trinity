import os
import re
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

class GenerationConfig(BaseModel):
    domain: str = Field(description="The data domain, either 'Healthcare' or 'Finance'")
    rows: int = Field(default=500, description="The number of records to generate")
    epsilon: float = Field(default=1.0, description="The privacy budget epsilon (0.1 for high privacy, 10.0 for high fidelity)")
    model: str = Field(default="auto", description="The generative model (auto, timegan, vae, diffusion)")
    columns: List[str] = Field(default=[], description="The list of column names or features to include")
    goal: str = Field(description="A concise summary of the synthesis goal")

    @validator("domain")
    def validate_domain(cls, v):
        if v.lower() not in ["healthcare", "finance"]:
            return "Healthcare" # Default fallback
        return v.capitalize()

class PromptParser:
    """
    Parses natural language prompts into structured GenerationConfig.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.parser = PydanticOutputParser(pydantic_object=GenerationConfig)
        
        if self.api_key:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert data engineer for AgentSynth-Trinity. Extract structured generation parameters from the user's prompt.\n{format_instructions}"),
                ("user", "{prompt}")
            ])
            self.chain = self.prompt_template | self.llm | self.parser

    def parse(self, user_prompt: str) -> GenerationConfig:
        """
        Parses the user prompt. Falls back to keyword extraction if API key is missing.
        """
        if not self.api_key:
            return self._fallback_parse(user_prompt)
        
        try:
            return self.chain.invoke({
                "prompt": user_prompt,
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            print(f"LLM Parsing Error: {e}")
            return self._fallback_parse(user_prompt)

    def _fallback_parse(self, prompt: str) -> GenerationConfig:
        """
        Basic regex/keyword extraction for when LLM is unavailable.
        """
        prompt_lower = prompt.lower()
        
        # Domain
        domain = "Healthcare"
        if "finance" in prompt_lower or "transaction" in prompt_lower or "fraud" in prompt_lower:
            domain = "Finance"
        elif "medical" in prompt_lower or "patient" in prompt_lower or "health" in prompt_lower:
            domain = "Healthcare"
            
        # Rows
        rows = 500
        row_match = re.search(r"(\d+)\s*(rows|records|samples)", prompt_lower)
        if row_match:
            rows = int(row_match.group(1))
            
        # Epsilon
        epsilon = 1.0
        if "high privacy" in prompt_lower or "strict" in prompt_lower:
            epsilon = 0.5
        elif "high fidelity" in prompt_lower or "accurate" in prompt_lower:
            epsilon = 5.0
            
        # Model
        model = "auto"
        if "timegan" in prompt_lower: model = "timegan"
        elif "vae" in prompt_lower: model = "vae"
        elif "diffusion" in prompt_lower: model = "diffusion"

        return GenerationConfig(
            domain=domain,
            rows=rows,
            epsilon=epsilon,
            model=model,
            goal=prompt[:100]
        )
