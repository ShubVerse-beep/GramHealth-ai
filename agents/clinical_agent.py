from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.config.settings import settings
import json

class ClinicalReasoningResponse(BaseModel):
    answer: str = Field(description="The textual clinical reasoning response provided by the agent.")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence level of the response.")
    requires_professional_review: bool = Field(description="Whether a doctor needs to review this situation.")
    grounded: bool = Field(description="True if the response is fully supported by the retrieved medical/patient evidence. False if there was no evidence or it was insufficient.")
    sources: List[str] = Field(description="List of chunk_ids that directly support the reasoning.", default=[])
    risk_level: Literal["low", "moderate", "high"] = Field(description="Assessed risk level.", default="low")
    recommended_next_step: str = Field(description="Recommended next step for the patient.", default="")

class ClinicalAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.0
        ).with_structured_output(ClinicalReasoningResponse)

    def reason(self, query: str, patient_evidence: List[Dict[str, Any]], medical_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        patient_context_str = json.dumps(patient_evidence, indent=2, ensure_ascii=False) if patient_evidence else "No patient context retrieved."
        medical_context_str = json.dumps(medical_evidence, indent=2, ensure_ascii=False) if medical_evidence else "No medical knowledge retrieved."
        
        prompt = f"""You are the Clinical Reasoning Layer.
Your responsibility is to analyze the user's query using ONLY the provided evidence.

PATIENT EVIDENCE:
{patient_context_str}

MEDICAL KNOWLEDGE EVIDENCE:
{medical_context_str}

IMPORTANT SAFETY RULES:
- Distinguish between what the patient's records say and what external medical knowledge says.
- Do not invent diagnoses, patient history, lab values, or unsupported facts.
- If the retrieved evidence does not contain sufficient information to answer safely, state that explicitly and set grounded=False.
- Recommend professional evaluation when appropriate.
- Cite the source chunk_ids in the 'sources' field if you relied on them.

User Query: {query}
"""
        result = self.llm.invoke(prompt)
        
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "requires_professional_review": result.requires_professional_review,
            "grounded": result.grounded,
            "sources": result.sources,
            "risk_level": result.risk_level,
            "recommended_next_step": result.recommended_next_step
        }
