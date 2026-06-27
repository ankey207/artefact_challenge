from __future__ import annotations

import re

from .config import DEFAULT_LIMIT

MAX_QUERY_LIMIT = 500
MAX_SQL_LENGTH = 12_000
MAX_JOINS = 6
MAX_CTES = 6
MAX_SUBQUERY_DEPTH = 8
MAX_STRING_LITERAL_LENGTH = 500

FORBIDDEN_PATTERNS = [
    r"\binsert\b",
    r"\bupdate\b",
    r"\bdelete\b",
    r"\bdrop\b",
    r"\balter\b",
    r"\bcreate\b",
    r"\bcopy\b",
    r"\battach\b",
    r"\bdetach\b",
    r"\binstall\b",
    r"\bload\b",
    r"\bpragma\b",
    r"\bcall\b",
    r"\bexport\b",
    r"\bimport\b",
    r"\brecursive\b",
    r"\bcross\s+join\b",
    r"\bunion\b",
    r"\bintersect\b",
    r"\bexcept\b",
]

ALLOWED_RELATIONS = {
    "circonscriptions",
    "candidats",
    "entity_aliases",
    "rag_chunks",
    "vw_results_clean",
    "vw_winners",
    "vw_turnout_by_region",
    "vw_national_summary",
}

# The analytics agent only needs this small function subset. An allowlist blocks
# resource-amplification and filesystem/system functions by default.
ALLOWED_FUNCTIONS = {
    "abs",
    "avg",
    "ceil",
    "coalesce",
    "count",
    "dense_rank",
    "floor",
    "greatest",
    "least",
    "lower",
    "max",
    "min",
    "nullif",
    "rank",
    "round",
    "row_number",
    "sum",
    "upper",
}

SQL_KEYWORDS_WITH_PARENS = {
    "and",
    "as",
    "case",
    "exists",
    "filter",
    "having",
    "in",
    "not",
    "on",
    "or",
    "over",
    "partition",
    "select",
    "using",
    "when",
    "where",
}


def strip_code_fences(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    return sql


def _sql_without_literals(sql: str) -> str:
    """Mask literals while preserving the surrounding SQL structure."""
    masked = re.sub(r"'(?:''|[^'])*'", "''", sql)
    return re.sub(r'"(?:""|[^"])*"', '""', masked)


def _validate_query_shape(cleaned: str) -> str:
    if len(cleaned) > MAX_SQL_LENGTH:
        return "SQL query is too long."
    if re.search(r"--|/\*|\*/", cleaned):
        return "SQL comments are not allowed."

    literals = re.findall(r"'((?:''|[^'])*)'", cleaned)
    if any(len(value) > MAX_STRING_LITERAL_LENGTH for value in literals):
        return "SQL string literal is too long."

    structural = _sql_without_literals(cleaned)
    lowered = structural.lower()
    if len(re.findall(r"\bjoin\b", lowered)) > MAX_JOINS:
        return "SQL query contains too many joins."
    if len(re.findall(r"\b[a-zA-Z_][\w]*\s+as\s*\(", lowered)) > MAX_CTES:
        return "SQL query contains too many CTEs."

    depth = current_depth = 0
    for char in structural:
        if char == "(":
            current_depth += 1
            depth = max(depth, current_depth)
        elif char == ")":
            current_depth -= 1
            if current_depth < 0:
                return "SQL parentheses are unbalanced."
    if current_depth != 0:
        return "SQL parentheses are unbalanced."
    if depth > MAX_SUBQUERY_DEPTH:
        return "SQL query is too deeply nested."

    functions = {match.group(1).lower() for match in re.finditer(r"\b([a-zA-Z_][\w]*)\s*\(", structural)}
    unsafe_functions = sorted(
        name for name in functions if name not in ALLOWED_FUNCTIONS and name not in SQL_KEYWORDS_WITH_PARENS
    )
    if unsafe_functions:
        return f"Unauthorized SQL function(s): {', '.join(unsafe_functions)}."
    return ""


def uppercase_norm_literals(sql: str) -> str:
    """Uppercase string literals compared against *_norm columns."""

    def replace_equals(match: re.Match[str]) -> str:
        return f"{match.group(1)}'{match.group(2).upper()}'"

    sql = re.sub(
        r"(\b\w+_norm\s*=\s*)'([^']*)'",
        replace_equals,
        sql,
        flags=re.IGNORECASE,
    )

    def replace_in(match: re.Match[str]) -> str:
        values = re.sub(
            r"'([^']*)'",
            lambda value_match: f"'{value_match.group(1).upper()}'",
            match.group(2),
        )
        return f"{match.group(1)}({values})"

    sql = re.sub(
        r"(\b\w+_norm\s+IN\s*)\(([^)]*)\)",
        replace_in,
        sql,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"(\b\w+_norm\s+(?:NOT\s+)?(?:I?LIKE)\s*)'([^']*)'",
        replace_equals,
        sql,
        flags=re.IGNORECASE,
    )


def validate_and_limit_sql(
    sql: str,
    default_limit: int = DEFAULT_LIMIT,
) -> tuple[bool, str, str]:
    """Validate generated SQL and return (ok, safe_sql, error_message)."""
    cleaned = uppercase_norm_literals(strip_code_fences(sql))
    cleaned = cleaned.strip().rstrip(";").strip()
    lowered = cleaned.lower()

    if not cleaned:
        return False, "", "No SQL query was generated."
    shape_error = _validate_query_shape(cleaned)
    if shape_error:
        return False, "", shape_error
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False, "", "Only SELECT/WITH queries are allowed."
    if ";" in cleaned:
        return False, "", "Multiple SQL statements are not allowed."
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, _sql_without_literals(cleaned), re.IGNORECASE):
            return False, "", "Unsafe SQL keyword detected."

    structural_lowered = _sql_without_literals(cleaned).lower()
    cte_names = {
        match.group(1).lower()
        for match in re.finditer(
            r"\bwith\s+([a-zA-Z_][\w]*)\s+as\b",
            structural_lowered,
        )
    }
    cte_names.update(
        match.group(1).lower()
        for match in re.finditer(
            r",\s*([a-zA-Z_][\w]*)\s+as\s*\(",
            structural_lowered,
        )
    )
    relations = {
        match.group(2).lower()
        for match in re.finditer(
            r"\b(from|join)\s+([a-zA-Z_][\w]*)",
            structural_lowered,
        )
    }
    unknown = sorted(
        relation for relation in relations if relation not in ALLOWED_RELATIONS and relation not in cte_names
    )
    if unknown:
        return (
            False,
            "",
            f"Query references unauthorized relation(s): {', '.join(unknown)}.",
        )

    if " limit " not in f" {lowered} ":
        cleaned = f"SELECT * FROM ({cleaned}) AS limited_result LIMIT {default_limit}"
    else:
        cleaned = re.sub(
            r"\bLIMIT\s+(\d+)",
            lambda match: f"LIMIT {min(int(match.group(1)), MAX_QUERY_LIMIT)}",
            cleaned,
            flags=re.IGNORECASE,
        )

    return True, cleaned, ""
