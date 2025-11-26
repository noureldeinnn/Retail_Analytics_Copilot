"""
SQLite tool for the Retail Analytics Copilot.

Responsibilities:
- Open the local Northwind SQLite database in data/northwind.sqlite.
- Provide a simple `run_query(sql)` function to execute SQL safely.
- Provide a `get_schema()` helper string for LLM prompts.
- Include a small manual test when run as a script.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Union
import sqlite3
import sys


# ---------- Locate and open the database ----------


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "northwind.sqlite"

if not DB_PATH.exists():
    
    print(f"[sqlite_tool] ERROR: Database not found at {DB_PATH}")
    print("Make sure you downloaded northwind.sqlite into the data/ folder.")
    
    

_CONN = sqlite3.connect(DB_PATH)
_CONN.row_factory = sqlite3.Row



def run_query(sql: str) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
    Execute a SQL query against the Northwind database.

    - On success:
        returns a list of dicts, one per row, with column names as keys.
        If the query doesn't return rows (e.g. UPDATE), returns an empty list.
    - On error:
        returns a dict: {"error": "<error-message>"}.

    This makes it easy for the agent logic to detect failures and trigger a
    repair loop.
    """
    cur = _CONN.cursor()
    try:
        cur.execute(sql)
    except sqlite3.Error as e:
        return {"error": str(e)}

    # If there is no result set (e.g. UPDATE/INSERT), description is None
    if cur.description is None:
        _CONN.commit()
        return []

    # Otherwise, build list of row dicts
    columns = [col[0] for col in cur.description]
    rows = cur.fetchall()
    result: List[Dict[str, Any]] = []
    for row in rows:
        row_dict: Dict[str, Any] = {}
        for col in columns:
            row_dict[col] = row[col]
        result.append(row_dict)
    return result


def get_schema() -> str:
    """
    Return a human-readable schema description for the main Northwind tables.

    This is not meant to be a perfect introspection; just enough for the LLM
    to understand table names and key columns when generating SQL.
    """
    # Canonical table names as described in the assignment
    # (even if you created lowercase views, we describe the originals here).
    schema_info = (
        "Tables and key columns:\n"
        "  Orders(\n"
        "    OrderID, CustomerID, EmployeeID, OrderDate, ShipCountry, ...\n"
        "  )\n"
        "  \"Order Details\"(\n"
        "    OrderID, ProductID, UnitPrice, Quantity, Discount\n"
        "  )\n"
        "  Products(\n"
        "    ProductID, ProductName, SupplierID, CategoryID, UnitPrice, ...\n"
        "  )\n"
        "  Customers(\n"
        "    CustomerID, CompanyName, ContactName, Country, ...\n"
        "  )\n"
        "  Categories(\n"
        "    CategoryID, CategoryName, Description\n"
        "  )\n"
        "  Suppliers(\n"
        "    SupplierID, CompanyName, Country, ...\n"
        "  )\n"
        "\n"
        "Optional lowercase views (if you created them):\n"
        "  orders, order_items, products, customers\n"
    )
    return schema_info


# ---------- Manual test ----------

if __name__ == "__main__":
    print("=== sqlite_tool manual test ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"DB path:      {DB_PATH}")

    if not DB_PATH.exists():
        print("\n[FAIL] northwind.sqlite not found. Did you put it in data/?")
        sys.exit(1)

    # List tables
    print("\n[1] Listing tables in the database:")
    cur = _CONN.cursor()
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()
    table_names = [row["name"] for row in tables]
    print("    Tables found:")
    for name in table_names:
        print(f"      - {name}")

    # Run a simple sample query
    print("\n[2] Running sample query: first 5 products")
    sample_sql = "SELECT ProductID, ProductName, UnitPrice FROM Products LIMIT 5;"
    result = run_query(sample_sql)

    if isinstance(result, dict) and "error" in result:
        print(f"    [FAIL] Error running sample query: {result['error']}")
        sys.exit(1)

    print("    Sample query result:")
    for row in result:
        print(f"      ProductID={row['ProductID']}, "
              f"Name={row['ProductName']}, "
              f"UnitPrice={row['UnitPrice']}")

    # Show schema helper
    print("\n[3] Schema helper string:")
    print(get_schema())


