from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from .state import GraphState
    from .nodes.select_loan_type import select_loan_type
    from .nodes.show_required_docs import show_required_docs
    from .nodes.upload_docs_and_form import upload_docs_and_form
    from .nodes.extract_documents import extract_documents
    from .nodes.check_doc_presence import check_doc_presence
    from .nodes.validate_conditions import validate_conditions
    from .nodes.score_threshold import score_threshold
    from .nodes.final_nodes import approve_loan, reject_loan, manual_review
    from .edges.routers import route_doc_check, route_final_decision
except ImportError:
    from state import GraphState
    from nodes.select_loan_type import select_loan_type
    from nodes.show_required_docs import show_required_docs
    from nodes.upload_docs_and_form import upload_docs_and_form
    from nodes.extract_documents import extract_documents
    from nodes.check_doc_presence import check_doc_presence
    from nodes.validate_conditions import validate_conditions
    from nodes.score_threshold import score_threshold
    from nodes.final_nodes import approve_loan, reject_loan, manual_review
    from edges.routers import route_doc_check, route_final_decision


def build_graph(use_checkpointer: bool = True):
    print("[GRAPH] 🏗️ Building StateGraph...")
    builder = StateGraph(GraphState)

    #   Register nodes  
    builder.add_node("select_loan_type",      select_loan_type)
    builder.add_node("show_required_docs",    show_required_docs)
    builder.add_node("upload_docs_and_form",  upload_docs_and_form)
    builder.add_node("extract_documents",     extract_documents)
    builder.add_node("check_doc_presence",    check_doc_presence)
    builder.add_node("validate_conditions",   validate_conditions)
    builder.add_node("score_threshold",       score_threshold)
    builder.add_node("approve_loan",          approve_loan)
    builder.add_node("manual_review",         manual_review)
    builder.add_node("reject_loan",           reject_loan)

    #   Entry point  
    builder.set_entry_point("select_loan_type")

    #   Linear edges  
    builder.add_edge("select_loan_type",     "show_required_docs")
    builder.add_edge("show_required_docs",   "upload_docs_and_form")
    builder.add_edge("upload_docs_and_form", "extract_documents")
    builder.add_edge("extract_documents",    "check_doc_presence")

    #   Conditional: doc presence check  
    builder.add_conditional_edges(
        "check_doc_presence",
        route_doc_check,
        {
            "missing_docs": END,
            "all_present":  "validate_conditions",
        },
    )

    builder.add_edge("validate_conditions", "score_threshold")

    builder.add_conditional_edges(
        "score_threshold",
        route_final_decision,
        {
            "approve":       "approve_loan",
            "manual_review": "manual_review",
            "reject":        "reject_loan",
        },
    )

    builder.add_edge("approve_loan",  END)
    builder.add_edge("manual_review", END)
    builder.add_edge("reject_loan",   END)

    checkpointer = MemorySaver() if use_checkpointer else None
    graph = builder.compile(checkpointer=checkpointer)
    print("[GRAPH] ✅ Compiled successfully.")
    return graph
