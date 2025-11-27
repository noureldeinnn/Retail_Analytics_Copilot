"""
Hybrid agent core: LangGraph-based RAG + SQL copilot (Baseline).

This file currently provides:
- Local LLM helper using Ollama + Phi-3.5.
- Router (docs | db | both).
- RAG retriever (from agent.rag.retrieval).
- Planner (extracts constraints from docs into text).
- Baseline NL->SQL using the LLM (no DSPy yet).
- SQL executor with a repair loop (up to 2 attempts).
- Synthesizer that uses docs + SQL result + format_hint to build an answer.
- LangGraph StateGraph wiring all of the above.
- Entry functions:
    - answer_question_entry(entry: dict) -> dict following Output Contract:

Output Contract (per question):
{
  "id": "...",
  "final_answer": <matches format_hint>,
  "sql": "<last executed SQL or empty if RAG-only>",
  "confidence": 0.0,
  "explanation": "<= 2 sentences>",
  "citations": [
    "Orders",
    "Order Details",
    "Products",
    "Customers",
    "kpi_definitions::chunk2",
    "marketing_calendar::chunk0"
  ]
}
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, TypedDict

import ollama
from langgraph.graph import StateGraph, START, END

# Import your local tools
from agent.tools import sqlite_tool
from agent.rag import retrieval

import os
from datetime import datetime


# ---------- LLM / Ollama helper ----------

MODEL_NAME = "phi3.5:3.8b-mini-instruct-q4_K_M"

def call_llm(prompt: str) -> str:
    """
    Call the local Phi-3.5 model via the official `ollama` Python client.
    """
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0.1,  # Low for deterministic results
            },
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama. Ensure '{MODEL_NAME}' is pulled and Ollama is running.\nError: {e}"
        )

    return response.get("response", "").strip()

# ---------- Tracing helper function ----------

def _save_trace_to_file(qid: str, state: AgentState, output: Dict[str, Any]) -> None:
    """
    Save a replayable trace for one question into logs/trace_..._<id>.log.
    This does NOT change the output contract, it's just side-effect logging.
    """
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_id = qid or "no_id"
    filename = f"logs/trace_{timestamp}_{safe_id}.log"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Retail Analytics Copilot Trace ===\n")
        f.write(f"ID: {safe_id}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("Question:\n")
        f.write(state.get("question", "") + "\n\n")

        f.write("Route:\n")
        f.write(str(state.get("route", "")) + "\n\n")

        f.write("Trace steps:\n")
        for step in state.get("trace", []):
            f.write(f"  - {step}\n")
        f.write("\n")

        f.write("Planner Notes:\n")
        f.write(str(state.get("planner_notes", "")) + "\n\n")

        f.write("SQL:\n")
        f.write(state.get("sql", "") + "\n\n")

        f.write("SQL Results:\n")
        f.write(str(state.get("sql_result", "")) + "\n\n")

        f.write("Final Answer:\n")
        f.write(str(output.get("final_answer")) + "\n\n")

        f.write("Citations:\n")
        f.write(", ".join(output.get("citations", [])) + "\n\n")

        f.write("Explanation:\n")
        f.write(output.get("explanation", "") + "\n\n")

    # Optional: small console hint
    print(f"[trace saved] {filename}")

# ---------- Router ----------

RouteType = Literal["docs", "db", "both"]


def route_question(question: str) -> RouteType:
    """
    Classify the user's intent based on the natural language question.
    """
    prompt = f"""
        You are the Retail Analytics Copilot for Northwind.

        Your task is to classify the question into exactly one category:
        - docs: Definitions, policies, returns, qualitative info that can be answered purely from docs.
        - db: Numeric aggregation, revenue, counts, top products, customer counts, etc. primarily from the DB.
        - both: Needs BOTH specific document constraints (dates/campaigns/KPIs) AND database aggregations.

        Now classify the following question.

        Question: {question}

        Return EXACTLY one word: docs, db, or both.
        """.strip()

    raw = call_llm(prompt).lower().strip()
    # Defensive parsing
    if "both" in raw:
        return "both"
    if "db" in raw:
        return "db"
    if "docs" in raw:
        return "docs"
    # Safe default fallback
    return "docs"


# ---------- NL -> SQL Generator ----------

def _clean_sql_output(raw_sql: str) -> str:
    """
    Extracts a usable SELECT statement from the model output.

    Simplified cleanup:
    - remove code fences
    - fix common 'Order Details' alias mistakes
    - ensure trailing semicolon
    """
    text = raw_sql.strip()

    # Remove markdown code fences
    if "```" in text:
        text = text.replace("```sql", "")
        text = text.replace("```", "")

    text = text.strip()

    # Fix common aliasing mistakes for "Order Details"
    text = text.replace("OrderDetails", "\"Order Details\"")
    text = text.replace("orderdetails", "\"Order Details\"")
    text = text.replace("Order_Details", "\"Order Details\"")
    text = text.replace("order_details", "\"Order Details\"")

    # Ensure it ends with ';'
    if not text.endswith(";"):
        text += ";"

    return text


def generate_sql(question: str, doc_context: str = "") -> str:
    """
    Generates a SQLite-compatible query using the schema + optional doc context.
    """
    schema = sqlite_tool.get_schema()

    context_part = f"\nContext from docs:\n{doc_context}\n" if doc_context else ""

    prompt = f"""
        You are the Retail Analytics Copilot for Northwind.
        Generate a valid SQLite query for the Northwind database that answers the question.

        Database schema (simplified):
        {schema}
        {context_part}

        Requirements:
        1. Use valid SQLite syntax. Use 'LIMIT n' for top results (NOT 'TOP n').
        2. Date handling: use strftime('%Y-%m-%d', column) when comparing to date strings.
        Example: strftime('%Y-%m-%d', OrderDate) BETWEEN '1997-06-01' AND '1997-06-30'.
        3. Revenue = SUM(UnitPrice * Quantity * (1 - Discount)).
        4. Tables: Orders, "Order Details", Products, Customers, Categories, Suppliers.
        5. Use proper JOINs (Orders -> "Order Details" -> Products, etc.).
        6. Return ONLY the SQL query. No explanation, no comments, no extra text.

        Question: {question}

        SQL:
        """.strip()

    raw_sql = call_llm(prompt)
    return _clean_sql_output(raw_sql)


# ---------- Agent State ----------

class AgentState(TypedDict, total=False):
    question: str
    format_hint: str

    route: RouteType
    planner_notes: str
    docs: List[Dict[str, Any]]

    sql: str
    sql_result: Any
    error: str

    final_answer_text: str
    explanation_text: str

    attempts: int
    trace: List[str]


def _append_trace(state: AgentState, message: str):
    trace = state.get("trace", [])
    trace.append(message)
    state["trace"] = trace


# ---------- Nodes ----------

def router_node(state: AgentState) -> AgentState:
    route = route_question(state["question"])
    state["route"] = route
    _append_trace(state, f"Router: Selected route='{route}'")
    return state


def retriever_node(state: AgentState) -> AgentState:
    """
    Retrieve docs if route is 'docs' or 'both'.
    """
    if state["route"] in ("docs", "both"):
        hits = retrieval.retrieve(state["question"], k=3)
        state["docs"] = hits
        _append_trace(state, f"Retriever: Found {len(hits)} chunks")
    else:
        state["docs"] = []
        _append_trace(state, "Retriever: Skipped (db-only route)")
    return state


def planner_node(state: AgentState) -> AgentState:
    """
    Extract constraints from docs (dates, KPI formulas, categories, entities).
    """
    docs = state.get("docs") or []
    if not docs:
        state["planner_notes"] = ""
        _append_trace(state, "Planner: No docs, skipping")
        return state

    docs_text = "\n".join([d["content"] for d in docs])

    prompt = f"""
        You are the Retail Analytics Copilot for Northwind.

        Extract key constraints (dates, specific categories, relevant KPIs/formulas)
        from the documentation that are relevant to answering this question.

        Question:
        {state['question']}

        Docs:
        {docs_text}

        Respond with a short bullet list of constraints.
        """.strip()

    state["planner_notes"] = call_llm(prompt)
    _append_trace(state, "Planner: Extracted constraints into planner_notes")
    return state


def nl2sql_node(state: AgentState) -> AgentState:
    """
    For 'db' or 'both' routes, generate SQL using the question + planner notes.
    For 'docs' route we should NOT even reach this node (graph handles that).
    """
    context = state.get("planner_notes", "")
    sql = generate_sql(state["question"], context)
    state["sql"] = sql
    _append_trace(state, f"SQL Gen: {sql}")
    return state


def executor_node(state: AgentState) -> AgentState:
    """
    Execute SQL against SQLite, with error handling compatible with sqlite_tool.

    sqlite_tool.run_query(sql) is expected to return either:
    - list[dict] on success
    - {"error": "..."} on failure
    """
    query = state.get("sql", "")
    if not query:
        state["error"] = "No SQL generated"
        state["sql_result"] = []
        _append_trace(state, "Executor: No SQL to execute")
        return state

    state["attempts"] = state.get("attempts", 0) + 1

    result = sqlite_tool.run_query(query)

    # Treat dict with {"error": "..."} as failure
    if isinstance(result, dict) and "error" in result:
        state["error"] = result["error"]
        state["sql_result"] = []
        _append_trace(state, f"Executor: Failed - {result['error']}")
    else:
        state["sql_result"] = result
        state["error"] = None
        n_rows = len(result) if isinstance(result, list) else 1
        _append_trace(state, f"Executor: Success ({n_rows} rows)")

    return state


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Synthesize final answer using:
    - question
    - docs
    - planner_notes
    - sql + sql_result
    - format_hint
    """
    sql_res = state.get("sql_result", [])
    sql_res_str = repr(sql_res)
    if len(sql_res_str) > 1000:
        sql_res_str = sql_res_str[:1000] + "...(truncated)"

    doc_texts = [d["content"] for d in state.get("docs", [])]

    prompt = f"""
        You are the Retail Analytics Copilot for Northwind.

        Question: {state['question']}
        Format Hint: {state['format_hint']}

        Data Context:
        - Planner Notes: {state.get("planner_notes", "None")}
        - SQL Query: {state.get("sql", "None")}
        - SQL Results: {sql_res_str}
        - Doc Excerpts: {doc_texts}

        TASK:
        1. Write FINAL_ANSWER that strictly matches the Format Hint:
        - If "int": return only digits (e.g., 14).
        - If "float": return a single number (e.g., 123.45).
        - If it looks like a JSON object schema (e.g., {{category:str, quantity:int}}),
            return a JSON object with those keys and types.
        - If it looks like "list[{{...}}]", return a JSON list of such objects.
        2. Write EXPLANATION (max 2 sentences) describing how you used docs and/or SQL.

        Output format (exactly):

        FINAL_ANSWER: <answer>
        EXPLANATION: <text>
        """.strip()

    raw = call_llm(prompt)

    # Parse model output into final_answer_text and explanation_text
    if "EXPLANATION:" in raw:
        parts = raw.split("EXPLANATION:", 1)
        state["final_answer_text"] = parts[0].replace("FINAL_ANSWER:", "").strip()
        state["explanation_text"] = parts[1].strip()
    else:
        state["final_answer_text"] = raw.strip()
        state["explanation_text"] = "Generated from context."

    _append_trace(state, "Synthesizer: Complete")
    return state


# ---------- Output Contract Helpers ----------

def _extract_final_answer_value(text: str, hint: str) -> Any:
    """
    Extract a properly-typed final_answer according to format_hint.

    Handles:
    - "int"
    - "float"
    - "{...}" (object schema) -> JSON object
    - "list[{...}]" (list-schema) -> JSON list
    - fallback: raw text
    """
    text = text.strip()
    hint = (hint or "").strip()

    # JSON-like formats
    is_list_schema = "list[" in hint.lower()
    is_object_schema = hint.startswith("{") and hint.endswith("}")

    if is_list_schema or is_object_schema:
        try:
            # Attempt to locate first JSON-ish block
            if "[" in text:
                start = text.find("[")
                end = text.rfind("]")
            elif "{" in text:
                start = text.find("{")
                end = text.rfind("}")
            else:
                start = end = -1

            if start != -1 and end != -1:
                json_str = text[start : end + 1]
                return json.loads(json_str)
        except Exception:
            # Fall through to numeric/text handling
            pass

    # Primitive numerics
    if hint == "int":
        nums = re.findall(r"-?\d+", text)
        return int(nums[0]) if nums else 0

    if hint == "float":
        nums = re.findall(r"-?\d+\.?\d*", text)
        return float(nums[0]) if nums else 0.0

    # Fallback: text
    return text


def _clip_explanation_to_two_sentences(explanation: str) -> str:
    explanation = explanation.strip()
    if not explanation:
        return ""
    # Naive sentence split on . ? !
    sentences = re.split(r"(?<=[.!?])\s+", explanation)
    return " ".join(sentences[:2]).strip()


def _extract_citations(state: AgentState) -> List[str]:
    """
    Build a citations list:
    - DB tables actually referenced in the SQL
    - doc chunk IDs used in RAG
    """
    citations: List[str] = []

    sql = state.get("sql") or ""
    sql_lower = sql.lower()

    # DB tables
    for table in ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"]:
        if table.lower() in sql_lower and table not in citations:
            citations.append(table)

    # Doc chunk IDs
    for d in state.get("docs") or []:
        cid = d.get("id")
        if cid and cid not in citations:
            citations.append(cid)

    return citations


def _estimate_confidence(error: str | None, attempts: int) -> float:
    """
    Simple heuristic confidence:
    - base 0.9
    - -0.2 if any error
    - -0.1 for each retry beyond the first
    Clamped to [0.1, 0.95].
    """
    conf = 0.9
    if error:
        conf -= 0.2
    if attempts > 1:
        conf -= 0.1 * (attempts - 1)
    return max(0.1, min(0.95, conf))


# ---------- Graph Construction ----------

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("router", router_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("planner", planner_node)
workflow.add_node("nl2sql", nl2sql_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)

# Edges
workflow.add_edge(START, "router")
workflow.add_edge("router", "retriever")
workflow.add_edge("retriever", "planner")


# CONDITIONAL ROUTING after planner:
# If it's pure docs, skip SQL and go directly to synthesizer.
def route_after_planner(state: AgentState) -> str:
    if state["route"] == "docs":
        return "synthesizer"
    return "nl2sql"


workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "synthesizer": "synthesizer",
        "nl2sql": "nl2sql",
    },
)

workflow.add_edge("nl2sql", "executor")


# REPAIR LOOP from executor:
def check_sql_error(state: AgentState) -> str:
    if state.get("error") and state.get("attempts", 0) < 2:
        return "nl2sql"  # Retry generating SQL
    return "synthesizer"


workflow.add_conditional_edges(
    "executor",
    check_sql_error,
    {
        "nl2sql": "nl2sql",
        "synthesizer": "synthesizer",
    },
)

workflow.add_edge("synthesizer", END)

app = workflow.compile()


# ---------- Public entrypoints ----------


def answer_question_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry used by your JSONL runner.

    Expects entry like those in sample_questions_hybrid_eval.jsonl:
    {
      "id": "...",
      "question": "...",
      "format_hint": "int" | "float" | "{...}" | "list[{...}]"
    }

    Returns Output Contract dict.
    """
    question = entry.get("question", "")
    format_hint = entry.get("format_hint", "text")
    qid = entry.get("id")

    inputs: AgentState = {
        "question": question,
        "format_hint": format_hint,
        "attempts": 0,
        "trace": [],
    }
    result: AgentState = app.invoke(inputs)

    final_answer = _extract_final_answer_value(result.get("final_answer_text", ""), format_hint)
    explanation = _clip_explanation_to_two_sentences(result.get("explanation_text", ""))
    citations = _extract_citations(result)
    confidence = _estimate_confidence(result.get("error"), result.get("attempts", 0))

    output = {
        "id": qid,
        "final_answer": final_answer,
        "sql": result.get("sql", "") or "",
        "confidence": confidence,
        "explanation": explanation,
        "citations": citations,
    }

    _save_trace_to_file(qid, result, output)

    return output

# ---------- Manual self-test ----------

if __name__ == "__main__":
    print("=== Testing Retail Analytics Copilot (RAG | SQL | Hybrid) ===\n")

    test_questions = [
        {
            "id": "rag_policy_beverages_return_days",
            "question": "According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.",
            "format_hint": "int",
        },
        {
            "id": "hybrid_top_category_qty_summer_1997",
            "question": "During 'Summer Beverages 1997' as defined in the marketing calendar, which product category had the highest total quantity sold? Return {category:str, quantity:int}.",
            "format_hint": "{category:str, quantity:int}",
        },
        {
            "id": "sql_top3_products_by_revenue_alltime",
            "question": "Top 3 products by total revenue all-time. Revenue uses Order Details: SUM(UnitPrice*Quantity*(1-Discount)). Return list[{product:str, revenue:float}].",
            "format_hint": "list[{product:str, revenue:float}]",
        },
    ]

    for t in test_questions:
        print(f"--- Testing: {t['id']} ---")
        print(f"Q: {t['question']}")
        out = answer_question_entry(t)
        print(out)
        print()
