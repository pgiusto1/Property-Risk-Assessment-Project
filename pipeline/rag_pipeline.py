import os
import json
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint

# Set Hugging Face API token
token = os.environ.get("HUGGINGFACE_TOKEN")
#from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    task="text-generation",  # ✅ this is required
    temperature=0.3
)
# Format structured data into a natural prompt
def build_prompt_from_inputs(flood_data: Dict, crime_data: Dict, address: str) -> str:
    prompt = f"""
Address: {address}

Flood Vulnerability:
- GEOID: {flood_data.get('GEOID')}
- FVI_storm_surge_2050s: {flood_data.get('FVI_storm_surge_2050s')}
- FVI_storm_surge_2080s: {flood_data.get('FVI_storm_surge_2080s')}
- FSHRI (Socioeconomic Vulnerability): {flood_data.get('FSHRI')}

Crime Statistics:
- Severity Score: {crime_data.get('severity_score')} per 1000 m²
- Borough Mean: {crime_data.get('mean')} per 1000 m²
- Std Dev: {crime_data.get('std')} per 1000 m²
- Area Sampled: {crime_data.get('area')} m²

Instructions:
Please assign a property risk score from 0–100, where:
- 0 = very low risk, 100 = very high risk
Explain how you weighed the flood and crime statistics. Highlight any features that influenced your judgment (e.g. future flood risk, below-average crime, etc.).
"""
    return prompt

def generate_risk_assessment(flood: Dict, crime: Dict, address: str):
    prompt_str = build_prompt_from_inputs(flood, crime, address)
    prompt = ChatPromptTemplate.from_template("""{input}""")
    chain = prompt | llm
    response = chain.invoke({"input": prompt_str})
    return response

# Example usage
if __name__ == "__main__":
    address = "621 Morgan Avenue, Brooklyn"

    flood_data = {
        "GEOID": "36061003400",
        "FVI_storm_surge_2050s": 2.0,
        "FVI_storm_surge_2080s": 2.0,
        "FSHRI": 1.0
    }

    crime_data = {
        "severity_score": 1.22,
        "mean": 2.00,
        "std": 1.15,
        "area": 507723.93
    }

    result = generate_risk_assessment(flood_data, crime_data, address)
    print("\n--- Final Risk Assessment (RAG with Mistral) ---")
    print(result)
