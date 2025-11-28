"""
run_agent_hybrid.py

CLI entrypoint for the Retail Analytics Copilot (hybrid RAG + SQL agent).

Contract (from assignment PDF):
We will run exactly:

    python run_agent_hybrid.py \
        --batch sample_questions_hybrid_eval.jsonl \
        --out outputs_hybrid.jsonl

- --batch: path to an input JSONL file.
  Each line is a JSON object with:
    {
      "id": "...",
      "question": "...",
      "format_hint": "int" | "float" | "{...}" | "list[{...}]"
    }

- --out: path to an output JSONL file.
  Each line must follow the Output Contract:
    {
      "id": "...",
      "final_answer": <matches format_hint>,
      "sql": "<last executed SQL or empty if RAG-only>",
      "confidence": 0.0,
      "explanation": "<= 2 sentences>",
      "citations": [ ... ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from agent.graph_hybrid import answer_question_entry

app = typer.Typer(help="Retail Analytics Copilot - Hybrid Agent CLI")


@app.command()
def run(
    batch: str = typer.Option(
        ...,
        "--batch",
        help="Path to input JSONL file (e.g., sample_questions_hybrid_eval.jsonl).",
    ),
    out: str = typer.Option(
        ...,
        "--out",
        help="Path to output JSONL file (e.g., outputs_hybrid.jsonl).",
    ),
) -> None:
    """
    Run the hybrid agent over a batch of questions in JSONL format
    and write outputs_hybrid.jsonl following the required Output Contract.
    """
    batch_path = Path(batch)
    out_path = Path(out)

    if not batch_path.exists():
        typer.echo(f"[ERROR] Batch file not found: {batch_path}")
        raise typer.Exit(code=1)

    typer.echo(f"=== Running Retail Analytics Copilot (hybrid) ===")
    typer.echo(f"Input batch : {batch_path}")
    typer.echo(f"Output file : {out_path}\n")

    num_inputs = 0
    num_outputs = 0

    # Make sure parent directory for output exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with batch_path.open("r", encoding="utf-8") as f_in, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            if not line.strip():
                continue

            num_inputs += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                typer.echo(f"[WARN] Skipping malformed JSON on line {num_inputs}: {e}")
                continue

            # Delegate to your LangGraph agent wrapper
            result = answer_question_entry(entry)

            # Ensure we always write one JSON object per line
            f_out.write(json.dumps(result, ensure_ascii=False))
            f_out.write("\n")
            num_outputs += 1

    typer.echo(f"\nDone.")
    typer.echo(f"Questions processed : {num_inputs}")
    typer.echo(f"Results written     : {num_outputs}")
    typer.echo(f"Output JSONL saved  : {out_path}")


if __name__ == "__main__":
    app()
