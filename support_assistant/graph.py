import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import mock_llm
import real_llm
import ingestion
from prompt_templates import SYSTEM_PROMPT_TEMPLATE

class GraphState(TypedDict):
    query: str
    intent: str
    context: List[dict]
    final_response: dict

def is_mock_mode() -> bool:
    mock_val = os.environ.get("MOCK_LLM", "1")
    return mock_val.strip() != "0"

def classify_intent_node(state: GraphState) -> GraphState:
    query = state["query"]
    if is_mock_mode():
        intent = mock_llm.mock_classify_intent(query)
    else:
        prompt = f"Classify this query as 'policy_question' or 'general_question'. Respond in JSON {{\"intent\": \"<intent>\"}}: Query: {query}"
        res = real_llm.call_real_llm(prompt)
        intent = res.get("intent", "general_question")
    
    return {**state, "intent": intent}

def retrieve_and_answer_node(state: GraphState) -> GraphState:
    query = state["query"]
    chunks = ingestion.query_vector_store(query, top_k=3)
    
    if is_mock_mode():
        resp = mock_llm.mock_retrieve_and_answer(query, chunks)
    else:
        formatted_context = "\n".join([f"[{c['id']}]: {c['text']}" for c in chunks])
        prompt = SYSTEM_PROMPT_TEMPLATE.format(context=formatted_context) + f"\nUser Query: {query}"
        resp = real_llm.real_llm_with_retry(prompt)
        
    return {**state, "context": chunks, "final_response": resp}

def direct_answer_node(state: GraphState) -> GraphState:
    if is_mock_mode():
        resp = mock_llm.mock_direct_answer()
    else:
        prompt = f"Answer directly as a customer service assistant: {state['query']}"
        resp = real_llm.real_llm_with_retry(prompt)
        
    return {**state, "final_response": resp}

def route_intent(state: GraphState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

workflow = StateGraph(GraphState)

workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("retrieve_and_answer", retrieve_and_answer_node)
workflow.add_node("direct_answer", direct_answer_node)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()