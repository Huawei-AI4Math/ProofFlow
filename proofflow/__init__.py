"""
ProofFlow: Automated Mathematical Proof Formalization using Large Language Models.

ProofFlow is a Python package that automatically converts natural language 
mathematical proofs into formalized Lean 4 code using Large Language Models (LLMs).
The package provides a complete pipeline for proof graph generation, formalization,
and automated proof generation.

Main Components:
    ProofFlow: Main class for proof formalization pipeline
    LLMManager: Manages LLM API calls and configurations
    LeanServer: Handles Lean 4 verification and code checking
    start_vllm_server: Utility for starting local vLLM servers

Example:
    >>> from proofflow import ProofFlow, LLMManager, LeanServer
    >>> 
    >>> # Set up components
    >>> lean_server = LeanServer(api_url="http://localhost:14457")
    >>> graph_model = LLMManager(model_info={...}, system_prompt_path="...")
    >>> 
    >>> # Initialize and use ProofFlow
    >>> proof_flow = ProofFlow(lean_server, graph_model, ...)
    >>> proof_flow.autoformalize_series("Theorem: For all x, x + 0 = x...")
    >>> lean_code = proof_flow.get_lean_code()
"""

from .lean_check import LeanServer
from .proofflow import ProofFlow
from .utils import LLMManager, start_vllm_server

# Version information
__version__ = "1.0.0"
__author__ = "ProofFlow Team"
__email__ = "contact@proofflow.ai"

# Main exports
__all__ = ["ProofFlow", "LLMManager", "LeanServer", "start_vllm_server"]
