
from typing import TypedDict, Optional, Literal


class GraphState(TypedDict, total=False):
    # Selection 
    loan_type: str                        
    required_docs: list[str]              # docs required for this loan type

    # Submission  

    uploaded_docs: dict[str, str]       
    form_data: dict                     

    # Extraction  
    extracted_docs: dict[str, str]        # doc_key → LLM-extracted text/JSON

    # Presence check  
    missing_docs: list[str]               # doc keys that are absent
    presence_passed: bool

    # Validation  
    # Per-doc result: {passed: bool, score: float 0-1, reason: str}
    validation_results: dict[str, dict]
    satisfaction_score: float             # mean of all doc scores (0.0 – 1.0)

    # Decision 
    final_decision: Optional[Literal["approve", "reject", "manual_review"]]
    rejection_reason: Optional[str]
    approval_message: Optional[str]
    review_notes: Optional[str]
    reasons: Optional[list[str]]

   
    error: Optional[str]
