import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import json5
from pydantic import BaseModel, Field, ValidationError

from .utils import LLMManager


class BaseTheoremComponent(BaseModel):
    id: str
    natural_language: str = Field(..., description="Exact text from theorem/proof")
    statement: str = Field(..., description="Self contained NL statement")
    dependencies: List[str] = Field(
        default_factory=list, description="List of dependency IDs"
    )
    formalization: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from formalization process"
    )
    solved_lemma: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from solver process"
    )
    score: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from scorer process"
    )
    error_report: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from error report"
    )


class TheoremCondition(BaseTheoremComponent):
    pass


class Definition(BaseTheoremComponent):
    pass


class Lemma(BaseTheoremComponent):
    # lean_hint: str = Field(..., description="Hint for proving this step")
    solved_negation: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from solver process of the negation."
    )


class TheoremStatement(BaseTheoremComponent):
    # lean_hint: str = Field(..., description="Final step hint")
    solved_negation: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from solver process of the negation."
    )


ProofGraphItem = TheoremCondition | Lemma | TheoremStatement


def check_DAG(validated_graph: List[ProofGraphItem]) -> List[ProofGraphItem]:
    """
    Checks and fixes the Directed Acyclic Graph (DAG) structure of the proof.

    1. Checks if lemmas are used as dependencies (warns if they're orphaned)
    2. checks if graph as cycles and forward references

    Args:
        validated_graph: List of validated proof graph items

    Returns:
        List of proof graph items with fixed dependencies
    """
    # Create a mapping of all item IDs for quick lookup
    all_ids = {item.id for item in validated_graph}

    # Track which lemma IDs are used as dependencies
    used_lemma_ids = set()

    # First pass: collect all dependencies to find used lemmas
    for item in validated_graph:
        used_lemma_ids.update(item.dependencies)

    # Second pass: fix empty dependencies and check for orphaned lemmas
    for i, item in enumerate(validated_graph):
        # Check if this is a lemma that's not used by any other item
        if not item.id.startswith("ts_") and item.id not in used_lemma_ids:
            error_msg = (
                f"'{item.id}' is not used as a dependency "
                f"by any subsequent lemma or theorem statement. "
                f"'{item.id}' needs to be used somewhere so that we have a valid proof."
                f"Please reconsider the graph structure."
            )
            print("Error: ", error_msg)
            raise ValueError(error_msg)

    # Third pass: validate DAG structure (check for cycles)
    def has_cycle(graph_items: List[ProofGraphItem]) -> bool:
        """Check if the dependency graph has any cycles."""
        # Build adjacency list
        adj_list = {item.id: item.dependencies for item in graph_items}

        # Track visit states: 0 = unvisited, 1 = visiting, 2 = visited
        visit_state = {item_id: 0 for item_id in all_ids}

        def dfs(node_id: str) -> bool:
            """Depth-first search to detect cycles."""
            if node_id not in adj_list:
                return False

            if visit_state[node_id] == 1:  # Currently visiting = cycle found
                return True
            if visit_state[node_id] == 2:  # Already visited
                return False

            visit_state[node_id] = 1  # Mark as visiting

            # Check all dependencies
            for dep_id in adj_list[node_id]:
                if dep_id in visit_state and dfs(dep_id):
                    return True

            visit_state[node_id] = 2  # Mark as visited
            return False

        # Check each node
        for item_id in all_ids:
            if visit_state[item_id] == 0:
                if dfs(item_id):
                    return True
        return False

    # Check for cycles
    if has_cycle(validated_graph):
        error_message = (
            "Cycle detected in the dependency graph! "
            "This violates the DAG property and will cause issues in proof formalization."
        )
        print("Error: ", error_msg)
        raise ValueError(error_message)

    # Additional validation: check for forward references
    for i, item in enumerate(validated_graph):
        available_ids = {validated_graph[j].id for j in range(i)}

        for dep_id in item.dependencies:
            if dep_id not in available_ids and dep_id not in all_ids:
                print(
                    f"Warning: Item '{item.id}' references unknown dependency '{dep_id}'"
                )
            elif dep_id not in available_ids:
                # This is a forward reference
                error_msg = (
                    f"Item '{item.id}' has a forward reference to '{dep_id}'. "
                    f"Dependencies should only reference previous items in the proof."
                )
                print("Error: ", error_msg)
                raise ValueError(error_msg)

    return validated_graph


def validate_proof_graph(data: List[dict]) -> List[ProofGraphItem]:
    """
    Validates the proof graph data using Pydantic models.
    Returns the validated data or raises ValidationError.
    """
    validated_items = []
    for item in data:
        if item["id"].startswith("ts"):
            validated_items.append(TheoremStatement(**item))
        elif item["id"].startswith("l"):
            validated_items.append(Lemma(**item))
        elif item["id"].startswith("def"):
            validated_items.append(Definition(**item))
        elif item["id"].startswith("tc"):
            validated_items.append(TheoremCondition(**item))
        else:
            raise ValueError(
                f"Unknown node type: {item['id']} -> Should be one of 'tc_', 'l_', 'def_', or 'ts_'"
            )
    return check_DAG(validated_items)


def extract_json_block(text: str):
    """Extract JSON block with multiple fallback strategies."""
    
    # Strategy 1: Look for fenced blocks (more permissive)
    patterns = [
        r"```json\s*\n?(.*?)\n?```",           # ```json ... ```
        r"```\s*\n?(.*?)\n?```",               # ``` ... ```  (no language)
        r"```json\s*(.*?)```",                 # ```json...``` (no newlines)
        r"```\s*(.*?)```"                      # ```...``` (no language, no newlines)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # Validate it looks like JSON
            if candidate.startswith(('{', '[')):
                return candidate
    
    # Strategy 2: Find balanced JSON object
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}' and start_idx != -1:
            brace_count -= 1
            if brace_count == 0:
                return text[start_idx:i+1]
    
    # Strategy 3: Look for any JSON-like structure
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(0)
        
    raise ValueError(f"No JSON block found. Text starts with: {repr(text[:100])}")

def sanitize_backslashes(raw: str) -> str:
    """
    Fix invalid JSON escape sequences by doubling backslashes.
    This version assumes any non-standard backslash is a single character
    that needs to be escaped. It correctly handles the LaTeX-style inputs.
    """

    # Replace common LaTeX sequences with their escaped equivalents
    raw = raw.replace('\\\\', '\\')  # Normalize any existing double backslashes
    raw = raw.replace('\\in', '\\\\in')
    raw = raw.replace('\\Q', '\\\\Q')
    # Let's assume the common LLM error is unescaped LaTeX commands.
    raw = raw.replace(r'\int', r'\\int')
    raw = raw.replace(r'\mathbb{Q}', r'\\mathbb{Q}')
    
    return raw.replace('\\', '\\\\')

def parse_llm_json(content: str):
    """Enhanced version with better debugging."""
    try:
        raw = extract_json_block(content)        
        # Try parsing without sanitization first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Only sanitize if regular parsing fails
            sanitized = sanitize_backslashes(raw)
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError:
                return json5.loads(sanitized)
                
    except Exception as e:
        print(f"DEBUG: Full content: {repr(content[:200])}...")  # Remove this in production
        raise ValueError(f"Failed to parse JSON: {e}") from e


def condition_on_all_previous_steps(graph_info: List[ProofGraphItem]):
    passed_lemmas = []
    for item in graph_info:
        item.dependencies = passed_lemmas.copy()  # Use copy to avoid reference issues
        passed_lemmas.append(item.id)


def build_proof_graph(
    natural_language_proof: str,
    model_manager: LLMManager,
    logs=None,
    follow_dag: bool = True,
    max_retries: int = 3,
) -> tuple[List[ProofGraphItem], int]:
    """
    Calls the LLM via call_llm_api with the proof_graph.md prompt as a system
    prompt and the provided natural language proof as user message.
    Returns the parsed and validated JSON proof graph as a list of Pydantic models.

    If validation fails, retries up to max_retries times with the original
    content and error message.
    """

    # system prompt added automatically
    messages = [{"role": "user", "content": natural_language_proof}]

    for attempt in range(max_retries):
        content, messages = model_manager.call_llm(messages, logs=logs)
        # Try to parse JSON
        try:
            proof_graph = parse_llm_json(content)
        except Exception as e:
            error_msg = "Could not parse proof graph JSON from the LLM output."
            messages.append(
                {"role": "user", "content": error_msg + ". Error: " + str(e)}
            )
            continue
        # Validate the parsed data
        try:
            validated_graph = validate_proof_graph(proof_graph)

            if not follow_dag:  # condition on all previous steps
                condition_on_all_previous_steps(validated_graph)
            return validated_graph, attempt + 1

        except (ValidationError, ValueError) as e:
            # Retry with error message
            error_msg = (
                f"JSON validation failed. Please fix the following errors: {str(e)} "
                f"Please provide a valid JSON structure that matches the expected format."
            )
            messages.append({"role": "user", "content": error_msg})

            print(str(e))

    # This should never be reached due to the exception handling above
    raise RuntimeError("Unexpected error in build_proof_graph")
