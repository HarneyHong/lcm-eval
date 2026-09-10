"""Comparable fingerprints for PostgreSQL text and JSON plans.

The paired collector executes the same SQL once in each EXPLAIN format.  A
query is admitted only when both executions produced the same physical plan.
The two formats spell several operators differently (for example
``Parallel Seq Scan`` versus ``Seq Scan`` plus ``Parallel Aware``), so both
representations are normalized before comparison.
"""

import copy
import re

from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_raw_plan


_OPERATOR_RE = re.compile(r"(?:->\s*)?(.*?)\s+\(cost=")
_INDEX_RE = re.compile(
    r"^(?P<node>.*?Index(?: Only)? Scan)(?: (?P<direction>Backward|Forward))?"
    r" using (?P<index>\S+) on (?P<relation>\S+)"
)
_BITMAP_INDEX_RE = re.compile(r"^(?P<node>Bitmap Index Scan) on (?P<index>\S+)")
_RELATION_RE = re.compile(r"^(?P<node>.*?Scan) on (?P<relation>\S+)")


def _clean_identifier(value):
    if value is None:
        return None
    value = str(value).strip('"')
    if "." in value:
        value = value.rsplit(".", 1)[-1].strip('"')
    return value


def _normalize_operator(description):
    parallel = description.startswith("Parallel ")
    if parallel:
        description = description[len("Parallel "):]

    partial_mode = None
    strategy = None
    for prefix, mode in (("Finalize ", "Finalize"), ("Partial ", "Partial")):
        if description.startswith(prefix):
            description = description[len(prefix):]
            partial_mode = mode
            break
    if description == "HashAggregate":
        description, strategy = "Aggregate", "Hashed"
    elif description == "GroupAggregate":
        description, strategy = "Aggregate", "Sorted"
    elif description == "Aggregate":
        strategy = "Plain"
    if description == "Aggregate" and partial_mode is None:
        partial_mode = "Simple"

    join_type = None
    for join_node in ("Hash Join", "Merge Join"):
        if description == join_node:
            join_type = "Inner"
            break
        join_prefix = join_node.rsplit(" ", 1)[0]
        join_suffix = join_node.rsplit(" ", 1)[1]
        match = re.fullmatch(rf"{re.escape(join_prefix)} (.+) {re.escape(join_suffix)}", description)
        if match:
            description = join_node
            join_type = match.group(1)
            break
    if description == "Nested Loop":
        join_type = "Inner"
    else:
        nested_match = re.fullmatch(r"Nested Loop (.+) Join", description)
        if nested_match:
            description = "Nested Loop"
            join_type = nested_match.group(1)

    return description, parallel, partial_mode, strategy, join_type


def _raw_node_fingerprint(node):
    operator_line = node.plain_content[0].strip()
    match = _OPERATOR_RE.search(operator_line)
    if match is None:
        raise ValueError(f"Could not find operator in raw plan line: {operator_line!r}")

    description = match.group(1).strip()
    relation = None
    index = None
    direction = None

    bitmap_index_match = _BITMAP_INDEX_RE.match(description)
    index_match = _INDEX_RE.match(description)
    if bitmap_index_match:
        description = bitmap_index_match.group("node")
        index = _clean_identifier(bitmap_index_match.group("index"))
    elif index_match:
        description = index_match.group("node")
        relation = _clean_identifier(index_match.group("relation"))
        index = _clean_identifier(index_match.group("index"))
        direction = index_match.group("direction") or "Forward"
    else:
        relation_match = _RELATION_RE.match(description)
        if relation_match:
            description = relation_match.group("node")
            relation = _clean_identifier(relation_match.group("relation"))

    node_type, parallel, partial_mode, strategy, join_type = _normalize_operator(description)
    return {
        "node_type": node_type,
        "parallel": parallel,
        "partial_mode": partial_mode,
        "strategy": strategy,
        "join_type": join_type,
        "relation": relation,
        "index": index,
        "scan_direction": direction,
        "children": [_raw_node_fingerprint(child) for child in node.children],
    }


def raw_plan_fingerprint(analyze_plan):
    root, _, _ = parse_raw_plan(copy.deepcopy(analyze_plan), analyze=True, parse=True)
    return _raw_node_fingerprint(root)


def _json_node_fingerprint(node):
    return {
        "node_type": node.get("Node Type"),
        "parallel": bool(node.get("Parallel Aware", False)),
        "partial_mode": node.get("Partial Mode"),
        "strategy": node.get("Strategy"),
        "join_type": node.get("Join Type"),
        "relation": _clean_identifier(node.get("Relation Name")),
        "index": _clean_identifier(node.get("Index Name")),
        "scan_direction": node.get("Scan Direction"),
        "children": [_json_node_fingerprint(child) for child in node.get("Plans", []) or []],
    }


def json_plan_fingerprint(analyze_plan):
    return _json_node_fingerprint(analyze_plan["Plan"])


def plans_are_equal(raw_analyze_plan, json_analyze_plan):
    return raw_plan_fingerprint(raw_analyze_plan) == json_plan_fingerprint(json_analyze_plan)
