"""
Evaluate NL→SQL AFTER DSPy optimization.

- Loads the whole DSPy program from nl2sql_prog/ (saved by nl2sql_dspy_train.py).
- Uses generate_sql_dspy(question, doc_context) to produce SQL.
- Executes against sqlite_tool.run_query(...) and computes:
    - valid-SQL rate
    - exec-success rate
- Logs detailed results into logs_after/.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict

import dspy

from agent.rag import retrieval
from agent.tools import sqlite_tool
from agent.graph_hybrid import _clean_sql_output


# ---------- 1. Configure DSPy LM (same as training) ----------

lm = dspy.LM(
    "ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M",
    api_base="http://localhost:11434",
    api_key="",
    temperature=0.1,
    max_tokens=512,
)
dspy.configure(lm=lm)


# ---------- 2. Paths ----------

HERE = os.path.dirname(__file__)
DATASET_PATH = os.path.join(HERE, "nl2sql_eval.jsonl")
MODEL_DIR = os.path.join(HERE, "nl2sql_prog")  # directory saved by prog.save(...)
LOGS_AFTER_DIR = os.path.join(HERE, "logs_after")
os.makedirs(LOGS_AFTER_DIR, exist_ok=True)


# ---------- 3. Load the saved whole program ----------

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        f"Could not find DSPy program directory at {MODEL_DIR}.\n"
        f"Run `python -m agent.nl2sql_eval.nl2sql_dspy_train` first."
    )

optimized_nl2sql = dspy.load(MODEL_DIR)  # per DSPy docs (whole-program load)


def generate_sql_dspy(question: str, doc_context: str = "") -> str:
    """
    Use the saved DSPy-optimized NL2SQL program to generate SQL.

    This mirrors the baseline generate_sql(...) interface.
    """
    schema = sqlite_tool.get_schema()
    pred = optimized_nl2sql(question=question, schema=schema, doc_context=doc_context)
    return _clean_sql_output(pred.sql)


# ---------- 4. Helpers for metrics ----------

def _classify_result(result: Any) -> Dict[str, Any]:
    """
    Turn sqlite_tool.run_query(...) result into flags + counts.
    """
    if isinstance(result, dict) and "error" in result:
        return {
            "is_valid": False,
            "is_exec_success": False,
            "rows_count": 0,
            "error": result["error"],
        }

    if isinstance(result, list):
        return {
            "is_valid": True,
            "is_exec_success": len(result) > 0,
            "rows_count": len(result),
            "error": "",
        }

    return {
        "is_valid": False,
        "is_exec_success": False,
        "rows_count": 0,
        "error": "Unknown result format",
    }


# ---------- 5. Main evaluation ----------

def evaluate_nl2sql_dspy(dataset_path: str = DATASET_PATH) -> None:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOGS_AFTER_DIR, f"nl2sql_dspy_{ts}.jsonl")

    total = 0
    valid_count = 0
    exec_count = 0

    print(f"=== NL→SQL eval AFTER DSPy on {dataset_path} ===")
    print(f"Using program from: {MODEL_DIR}")
    print(f"Logging detailed results to: {log_path}\n")

    with open(dataset_path, "r", encoding="utf-8") as f_in, \
         open(log_path, "w", encoding="utf-8") as f_log:

        for line in f_in:
            if not line.strip():
                continue

            entry = json.loads(line)
            qid = entry.get("id", f"q{total+1}")
            question = entry.get("question", "")

            total += 1
            print(f"[{total:02d}] {qid}: {question}")

            # 1) Optional doc_context, to mirror baseline behavior
            hits = retrieval.retrieve(question, k=3)
            doc_context = "\n".join(h["content"] for h in hits)

            # 2) Generate SQL with saved DSPy program
            sql = generate_sql_dspy(question, doc_context)

            # 3) Execute SQL on SQLite
            result = sqlite_tool.run_query(sql)
            stats = _classify_result(result)

            if stats["is_valid"]:
                valid_count += 1
            if stats["is_exec_success"]:
                exec_count += 1

            print(
                f"    -> valid={stats['is_valid']}, "
                f"exec_success={stats['is_exec_success']}, "
                f"rows={stats['rows_count']}"
            )
            if stats["error"]:
                print(f"    -> error: {stats['error']}")
            print(f"    -> SQL: {sql}\n")

            # 4) Log full info
            log_entry = {
                "id": qid,
                "question": question,
                "sql": sql,
                "is_valid": stats["is_valid"],
                "is_exec_success": stats["is_exec_success"],
                "rows_count": stats["rows_count"],
                "error": stats["error"],
            }
            f_log.write(json.dumps(log_entry) + "\n")

    if total == 0:
        print("No questions found in dataset. Nothing to evaluate.")
        return

    valid_rate = valid_count / total
    exec_rate = exec_count / total

    print("=== NL→SQL metrics AFTER DSPy ===")
    print(f"Total questions:           {total}")
    print(f"Valid-SQL count / rate:    {valid_count} / {valid_rate:.2f}")
    print(f"Exec-success count / rate: {exec_count} / {exec_rate:.2f}")
    print("\nCompare to your baseline metrics for before/after.\n")


if __name__ == "__main__":
    evaluate_nl2sql_dspy()
