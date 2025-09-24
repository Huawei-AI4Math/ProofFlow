#!/usr/bin/env python3
"""
Command-line interface for ProofFlow.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .lean_check import LeanServer
from .proofflow import ProofFlow
from .utils import LLMManager


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="ProofFlow: Automated mathematical proof formalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single proof from command line
  proofflow --proof "Theorem: For all x, x + 0 = x. Proof: By definition of addition."

  # Process proofs from a JSON file
  proofflow --input data/proofs.json --output results/

  # Use custom model configuration
  proofflow --proof "..." --model-config config.json

  # Process with specific Lean server
  proofflow --proof "..." --lean-server http://localhost:14457
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--proof", type=str, help="Natural language proof to formalize"
    )
    input_group.add_argument(
        "--input", type=str, help="Path to input file (JSON or Excel)"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results (default: current directory)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "pickle", "lean"],
        default="json",
        help="Output format (default: json)",
    )

    # Model configuration
    parser.add_argument(
        "--model-config", type=str, help="Path to model configuration JSON file"
    )
    parser.add_argument("--api-key", type=str, help="API key for LLM service")
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for LLM API (default: OpenRouter)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4",
        help="LLM model to use (default: claude-sonnet-4)",
    )

    # Lean configuration
    parser.add_argument(
        "--lean-server", type=str, help="Lean server URL (default: local project)"
    )
    parser.add_argument("--lean-project", type=str, help="Path to local Lean project")

    # Processing options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for each step (default: 3)",
    )
    parser.add_argument(
        "--follow-dag",
        action="store_true",
        default=True,
        help="Follow DAG structure (default: True)",
    )

    return parser


def load_model_config(config_path: str) -> dict:
    """Load model configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def create_llm_managers(args) -> tuple:
    """Create LLM managers based on arguments."""
    if args.model_config:
        config = load_model_config(args.model_config)
        graph_config = config.get("graph_model", {})
        formalize_config = config.get("formalize_model", {})
        solver_config = config.get("solver_model", {})
    else:
        # Default configuration
        base_config = {
            "api_key": args.api_key,
            "base_url": args.base_url,
            "model": args.model,
        }
        graph_config = base_config.copy()
        formalize_config = base_config.copy()
        solver_config = base_config.copy()

    graph_model = LLMManager(
        model_info=graph_config, system_prompt_path="prompts/proof_graph.md"
    )

    formalize_model = LLMManager(
        model_info=formalize_config, system_prompt_path="prompts/lemma_formalizer.md"
    )

    solver_model = LLMManager(
        model_info=solver_config, system_prompt_path="prompts/lemma_prover.md"
    )

    return graph_model, formalize_model, solver_model


def create_lean_server(args) -> LeanServer:
    """Create Lean server based on arguments."""
    if args.lean_server:
        return LeanServer(api_url=args.lean_server)
    elif args.lean_project:
        return LeanServer(project_path=args.lean_project)
    else:
        # Try to find a local Lean project
        current_dir = Path.cwd()
        if (current_dir / "lakefile.lean").exists():
            return LeanServer(project_path=str(current_dir))
        else:
            raise ValueError(
                "No Lean server or project specified. Use --lean-server or --lean-project."
            )


def process_single_proof(proof: str, args) -> dict:
    """Process a single proof."""
    # Create managers
    graph_model, formalize_model, solver_model = create_llm_managers(args)
    lean_server = create_lean_server(args)

    # Create proof flow
    proof_flow = ProofFlow(
        lean_server=lean_server,
        graph_model_manager=graph_model,
        formalize_model_manager=formalize_model,
        solver_model_manager=solver_model,
        verbose=args.verbose,
    )

    # Process proof
    proof_flow.autoformalize_series(
        proof,
        graph_builder_retries=args.retries,
        formalizer_retries=args.retries,
        prover_retries=args.retries,
        follow_dag=args.follow_dag,
    )

    # Get results
    lean_code = proof_flow.get_lean_code()
    summary = proof_flow.summary()

    return {
        "lean_code": lean_code,
        "summary": summary,
        "proof_items": [item.model_dump() for item in proof_flow.proof_items],
    }


def process_batch_input(input_path: str, args) -> list:
    """Process batch input from file."""
    input_file = Path(input_path)

    if input_file.suffix == ".json":
        with open(input_file, "r") as f:
            data = json.load(f)
    elif input_file.suffix in [".xlsx", ".xls"]:
        import pandas as pd

        df = pd.read_excel(input_file)
        data = []
        for _, row in df.iterrows():
            if "theorem" in row and "proof" in row:
                data.append(
                    {"theorem": str(row["theorem"]), "proof": str(row["proof"])}
                )
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")

    results = []
    for i, item in enumerate(data):
        if args.verbose:
            print(f"Processing item {i + 1}/{len(data)}")

        if "theorem" in item and "proof" in item:
            proof_text = f"Theorem: {item['theorem']}\nProof: {item['proof']}"
        else:
            proof_text = str(item)

        result = process_single_proof(proof_text, args)
        result["input_item"] = item
        results.append(result)

    return results


def save_results(results: list, output_dir: str, format: str):
    """Save results to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if len(results) == 1:
        # Single result
        result = results[0]
        if format == "lean":
            with open(output_path / "proof.lean", "w") as f:
                f.write(result["lean_code"])
        elif format == "json":
            with open(output_path / "result.json", "w") as f:
                json.dump(result, f, indent=2)
        elif format == "pickle":
            import pickle

            with open(output_path / "result.pkl", "wb") as f:
                pickle.dump(result, f)
    else:
        # Multiple results
        for i, result in enumerate(results):
            if format == "lean":
                with open(output_path / f"proof_{i}.lean", "w") as f:
                    f.write(result["lean_code"])
            elif format == "json":
                with open(output_path / f"result_{i}.json", "w") as f:
                    json.dump(result, f, indent=2)
            elif format == "pickle":
                import pickle

                with open(output_path / f"result_{i}.pkl", "wb") as f:
                    pickle.dump(result, f)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.proof:
            # Single proof processing
            results = [process_single_proof(args.proof, args)]
        else:
            # Batch processing
            results = process_batch_input(args.input, args)

        # Save results
        output_dir = args.output or "."
        save_results(results, output_dir, args.format)

        if args.verbose:
            print(f"Results saved to {output_dir}")
            if len(results) == 1:
                summary = results[0]["summary"]
                print(f"Formalization accuracy: {summary['form_acc']:.2%}")
                print(f"Proof success rate: {summary['solv_acc']:.2%}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
