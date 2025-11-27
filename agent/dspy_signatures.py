# agent/dspy_signatures.py


import dspy

class NL2SQLSignature(dspy.Signature):
    """
    Given a natural language question, the database schema, and optional document
    context (date ranges, categories, KPI formulas), generate a SINGLE valid
    SQLite SELECT query for the Northwind database.

    Requirements:
    - Use only tables and columns that exist in the provided schema.
    - Use correct table names (e.g., "Order Details" with quotes).
    - Return exactly one SELECT statement, terminated with ';'.
    - Do NOT include explanations, comments, or any text before/after the SQL.
    """
    question = dspy.InputField(desc="Natural language analytics question about the Northwind dataset.")
    schema = dspy.InputField(desc="SQLite schema of the Northwind database as plain text.")
    doc_context = dspy.InputField(desc="Optional marketing calendar or KPI definitions relevant to the question.")
    sql = dspy.OutputField(desc="A single valid SQLite SELECT query with a trailing ';' and no extra text.")


class NL2SQLModule(dspy.Module):
    """
    Small DSPy module that wraps NL2SQLSignature.
    We'll optimize this with BootstrapFewShot or MIPROv2.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(NL2SQLSignature)

    def forward(self, question: str, schema: str, doc_context: str = ""):
        return self.predict(question=question, schema=schema, doc_context=doc_context)
