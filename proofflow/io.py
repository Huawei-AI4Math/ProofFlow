"""
I/O utilities for saving and loading ProofFlow state.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from . import ProofFlow

class RenamedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Check if the module name starts with the old root name
        if module.startswith('autoformalize'):
            # Construct the new module path by replacing the old root
            new_module = 'proofflow' + module[len('autoformalize'):]
            
            # Use importlib to dynamically find and load the class from the new module
            try:
                __import__(new_module)
                new_class = getattr(sys.modules[new_module], name)
                return new_class
            except (ImportError, AttributeError):
                # If the new class doesn't exist or can't be imported,
                # fall back to the default unpickler
                pass

        # For all other classes, use the default unpickler behavior
        return super().find_class(module, name)

def save_proofflow(
    self: "ProofFlow", filepath: str, format: str = "pickle"
) -> None:
    """
    Save the essential fields of the ProofFlow instance to a file.

    Args:
        filepath: Path where to save the data
        format: 'pickle' or 'json' (default: 'pickle')
                Note: 'json' requires proof_items to be serializable
    """
    data_to_save = {
        "proof_items": self.proof_items,
        "nl_proof": self.nl_proof,
        "llm_call_logs": self.llm_call_logs,
    }

    filepath = Path(filepath)

    if format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(data_to_save, f)
        if self.verbose:
            self._print_status(
                f"Saved proofflow state to {filepath} (pickle format)",
                style="okgreen",
            )

    elif format == "json":
        # For JSON, we need to convert proof_items to dictionaries
        json_data = {"nl_proof": self.nl_proof, "llm_call_logs": self.llm_call_logs, "total_score": self.total_score}

        # Convert proof_items to dictionaries if they exist
        if self.proof_items is not None:
            json_data["proof_items"] = [
                item.model_dump() if hasattr(item, "model_dump") else item.__dict__
                for item in self.proof_items
            ]
        else:
            json_data["proof_items"] = None

        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=2)
        if self.verbose:
            self._print_status(
                f"Saved proofflow state to {filepath} (JSON format)",
                style="okgreen",
            )

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'")


def load_proofflow(
    self: "ProofFlow", filepath: str, format: str = "pickle"
) -> None:
    """
    Load the essential fields into the ProofFlow instance from a file.

    Args:
        filepath: Path from where to load the data
        format: 'pickle' or 'json' (default: 'pickle')
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if format == "pickle":
        with open(filepath, "rb") as f:
            unpickler = RenamedUnpickler(f)
            data = unpickler.load()

        self.proof_items = data.get("proof_items")
        self.nl_proof = data.get("nl_proof")
        self.llm_call_logs = data.get("llm_call_logs", [])

    elif format == "json":
        with open(filepath, "r") as f:
            data = json.load(f)

        # For JSON, proof_items will be dictionaries that need to be reconstructed
        # This assumes you have a way to reconstruct the objects from dictionaries
        self.proof_items = data.get("proof_items")
        self.nl_proof = data.get("nl_proof")
        self.llm_call_logs = data.get("llm_call_logs", [])

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'")

    # Rebuild the graph if proof_items exist
    if self.proof_items is not None:
        try:
            from .vis import build_dag

            self.graph, self._node_info = build_dag(
                [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in self.proof_items
                ]
            )
        except Exception as e:
            if self.verbose:
                self._print_status(f"Could not rebuild graph: {e}", style="warning")


def load_proofflow_from_file(
    cls,
    filepath: str,
    lean_server,
    graph_model_manager,
    formalize_model_manager,
    solver_model_manager,
    score_model_manager=None,
    verbose=True,
    format="pickle",
):
    """
    Class method to create a new ProofFlow instance from a saved file.

    Args:
        filepath: Path from where to load the data
        lean_server: LeanServer instance
        graph_model_manager: LLMManager for graph building
        formalize_model_manager: LLMManager for formalization
        solver_model_manager: LLMManager for solving
        score_model_manager: Optional LLMManager for scoring
        verbose: Whether to print status messages
        format: 'pickle' or 'json' (default: 'pickle')

    Returns:
        ProofFlow instance with loaded state
    """
    instance = cls(
        lean_server=lean_server,
        graph_model_manager=graph_model_manager,
        formalize_model_manager=formalize_model_manager,
        solver_model_manager=solver_model_manager,
        score_model_manager=score_model_manager,
        verbose=verbose,
    )
    load_proofflow(instance, filepath, format)
    return instance
