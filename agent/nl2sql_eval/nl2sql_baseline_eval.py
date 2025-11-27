"""
Baseline NL→SQL evaluation script.

This script DOES NOT run the full LangGraph agent.
It only measures how good the current generate_sql(...) is at:

- producing valid SQLite queries (no errors),
- producing queries that actually return rows (exec success),

on a small set of natural language SQL questions stored in nl2sql_eval.jsonl.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict

from agent.rag import retrieval
from agent.tools import sqlite_tool
from agent.graph_hybrid import generate_sql  # baseline NL→SQL


# Path setup
HERE = os.path.dirname(__file__)
DATASET_PATH = os.path.join(HERE, "nl2sql_eval.jsonl")

LOGS_BEFORE_DIR = os.path.join(HERE, "logs_before")
os.makedirs(LOGS_BEFORE_DIR, exist_ok=True)


def _is_valid_and_exec_success(result: Any) -> Dict[str, Any]:
    """
    Given sqlite_tool.run_query(result), classify into:
      - is_valid: no SQL/exec error
      - is_exec_success: valid AND returns at least one row
      - rows_count: how many rows (0 if not a list)
      - error: error message if any
    """
    is_valid = True
    is_exec_success = False
    rows_count = 0
    error = ""

    # sqlite_tool convention: dict with "error" on failure, list[dict] on success
    if isinstance(result, dict) and "error" in result:
        is_valid = False
        error = result["error"]
    elif isinstance(result, list):
        rows_count = len(result)
        is_exec_success = rows_count > 0
    else:
        # Unexpected type, treat as valid but no rows
        rows_count = 0
        is_exec_success = False

    return {
        "is_valid": is_valid,
        "is_exec_success": is_exec_success,
        "rows_count": rows_count,
        "error": error,
    }


def evaluate_nl2sql_baseline(dataset_path: str = DATASET_PATH) -> None:
    """
    Measure baseline NL→SQL performance using the current generate_sql(...)
    and sqlite_tool.run_query(...).

    Metrics:
      - valid-SQL rate  = fraction of queries that do not error
      - exec-success    = fraction of queries that return at least 1 row
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Prepare log file
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOGS_BEFORE_DIR, f"nl2sql_baseline_{ts}.jsonl")

    total = 0
    valid_count = 0
    exec_count = 0

    print(f"=== Baseline NL→SQL eval on {dataset_path} ===\n")
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

            # 1) Retrieve docs to simulate planner context (optional but realistic)
            hits = retrieval.retrieve(question, k=3)
            doc_context = "\n".join(h["content"] for h in hits)

            # 2) Generate SQL with current baseline generator
            sql = generate_sql(question, doc_context)

            # 3) Execute SQL
            result = sqlite_tool.run_query(sql)
            stats = _is_valid_and_exec_success(result)

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

            # 4) Log detailed info as JSONL for later analysis
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

    print("=== Baseline NL→SQL metrics ===")
    print(f"Total questions:           {total}")
    print(f"Valid-SQL count / rate:    {valid_count} / {valid_rate:.2f}")
    print(f"Exec-success count / rate: {exec_count} / {exec_rate:.2f}")
    print("\nUse these as your BEFORE metrics for DSPy optimization.\n")


if __name__ == "__main__":
    evaluate_nl2sql_baseline()
