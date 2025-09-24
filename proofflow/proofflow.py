from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from .lean_check import LeanServer
from .proof_formalize import run_formalizer_prompt
from .proof_graph import Definition, TheoremCondition, build_proof_graph
from .proof_prover import run_solver_prompt
from .proof_scorer import compute_total_score, run_node_scorer_prompt
from .utils import LLMManager, remove_imports
from .vis import (
    build_dag,
    create_interactive_visualization,
    create_static_visualization,
)


class ProofFlow:
    """
    Main class for automated mathematical proof formalization using Large Language Models.
    
    ProofFlow converts natural language mathematical proofs into formalized Lean 4 code
    through a multi-step process involving proof graph generation, formalization,
    and automated proof generation.
    
    Attributes:
        lean_server (LeanServer): Server for Lean 4 verification
        graph_model_manager (LLMManager): LLM for proof graph generation
        formalize_model_manager (LLMManager): LLM for natural language formalization
        solver_model_manager (LLMManager): LLM for automated proof generation
        score_model_manager (Optional[LLMManager]): LLM for semantic scoring
        verbose (bool): Whether to print progress information
        proof_items (Optional[List]): Generated proof graph items
        nl_proof (Optional[str]): Original natural language proof
        llm_call_logs (List[Dict]): Logs of all LLM API calls
    """
    def __init__(self, 
                 lean_server: LeanServer,
                 graph_model_manager: LLMManager,
                 formalize_model_manager: LLMManager,
                 solver_model_manager: LLMManager,
                 score_model_manager: Optional[LLMManager] = None,
                 verbose: bool = True):
        """
        Initialize ProofFlow with required components.
        
        Args:
            lean_server: Server for Lean 4 verification
            graph_model_manager: LLM manager for proof graph generation
            formalize_model_manager: LLM manager for natural language formalization
            solver_model_manager: LLM manager for automated proof generation
            score_model_manager: Optional LLM manager for semantic scoring
            verbose: Whether to print progress information during processing
        """
        self.proof_items = None
        self.nl_proof = None
        self.verbose = verbose
        self.graph_model_manager = graph_model_manager
        self.formalize_model_manager = formalize_model_manager 
        self.solver_model_manager = solver_model_manager 
        self.score_model_manager = score_model_manager 
        self.llm_call_logs = []  # Store LLM call logs per instance
        self.lean_server = lean_server
    
    # ===== PRIVATE UTILITIES =====
    
    def _print_status(self, msg: str, style: Optional[str] = None) -> None:
        """
        Print status message with optional styling if verbose mode is enabled.
        
        Args:
            msg: Message to print
            style: Optional style name ('header', 'okblue', 'okgreen', 'warning', 'fail', 'bold', 'underline')
        """
        if not self.verbose:
            return
        # ANSI color codes for style
        styles = {
            'header': '\033[95m',
            'okblue': '\033[94m',
            'okgreen': '\033[92m',
            'warning': '\033[93m',
            'fail': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m',
            'end': '\033[0m',
        }
        if style and style in styles:
            print(f"{styles[style]}{msg}{styles['end']}")
        else:
            print(msg)

    def _print_progress_summary(self, tries: int) -> None:
        """
        Print summary of proof graph generation progress.
        
        Args:
            tries: Number of attempts made during graph generation
        """
        n_conditions = sum(1 for item in self.proof_items if item.id.startswith('tc_'))
        n_lemmas = sum(1 for item in self.proof_items if item.id.startswith('l'))
        n_solutions = sum(1 for item in self.proof_items if item.id.startswith('ts_'))
        if tries:
            self._print_status(f"\nProof graph completed ({tries} tries): {n_conditions} condition(s), {n_lemmas} lemma(s), {n_solutions} theorem solution(s).", style='header')
        else:
            self._print_status(f"{n_conditions} condition(s), {n_lemmas} lemma(s), {n_solutions} theorem solution(s).", style='header')

    def _print_item_progress(self, idx: int, total: int, item_id: str, completed: bool = False) -> None:
        """
        Print progress for individual proof item processing.
        
        Args:
            idx: Current item index (0-based)
            total: Total number of items
            item_id: Unique identifier for the item
            completed: Whether the item has been completed
        """
        if completed:
            self._print_status(f"   \u2714 Completed item {item_id}.", style='okgreen')
        else:
            self._print_status(f"\u27A4 Formalizing item {idx+1}/{total}: {item_id} ...", style='okblue')

    # ===== CORE FORMALIZATION =====
    
    def autoformalize_series(self, nl_proof: str,
                             graph_builder_retries: int = 3, 
                             formalizer_retries: int = 3,
                             prover_retries: int = 3,
                             follow_dag: bool = True,
                             previous_context: bool = True,
                             supply_proof: bool = True) -> None:
        """
        Process a natural language proof through the complete formalization pipeline.
        
        This method performs the main proof formalization workflow:
        1. Builds a proof graph from the natural language proof
        2. Formalizes each proof step into Lean 4 code
        3. Attempts to automatically prove each formalized step
        
        Args:
            nl_proof: Natural language proof text to formalize
            graph_builder_retries: Number of retries for proof graph generation
            formalizer_retries: Number of retries for each formalization step
            prover_retries: Number of retries for each proof generation step
            follow_dag: Whether to follow DAG structure or condition on all previous steps
            previous_context: Whether to provide dependency statements during formalization
            supply_proof: Whether to supply original proof text at each step for context
        """
        self.nl_proof = nl_proof
        
        self._print_status("\nBuilding proof graph...", style='bold')

        #Step 1. Build proof graph
        self.proof_items, tries = build_proof_graph(nl_proof, 
                                                    model_manager = self.graph_model_manager, 
                                                    logs = self.llm_call_logs, 
                                                    follow_dag = follow_dag,
                                                    max_retries = graph_builder_retries)
        self._print_progress_summary(tries)

        for idx, item in enumerate(self.proof_items):
            self._print_item_progress(idx, len(self.proof_items), item.id)

            #Step 1. Formalize statement with Goedel-Formalizer
            formalization  = run_formalizer_prompt(item, 
                                                   lean_server = self.lean_server,
                                                   all_items = self.proof_items, 
                                                   model_manager = self.formalize_model_manager, 
                                                   logs = self.llm_call_logs,
                                                   max_retries = formalizer_retries,
                                                   previous_context = previous_context,
                                                   original_proof = self.nl_proof if supply_proof else "")
            item.formalization = formalization

            #Step 2. Solver with tactics
            solved_lemma = run_solver_prompt(item, 
                                             lean_server=self.lean_server,
                                             model_manager=self.solver_model_manager, 
                                             logs=self.llm_call_logs,
                                             max_retries=prover_retries)
            item.solved_lemma = solved_lemma
        
            self._print_item_progress(idx, len(self.proof_items), item.id, completed=True)

    # ===== DATA ACCESS =====
    
    def get_llm_call_logs(self) -> List[Dict[str, Any]]:
        """
        Get logs of all LLM API calls made during processing.
        
        Returns:
            List of dictionaries containing call details (model, tokens, timing, etc.)
        """
        return self.llm_call_logs

    # ===== CODE GENERATION =====
    
    def get_lean_code(self) -> str:
        """
        Generate complete Lean 4 code from formalized proof items.
        
        Collects Lean code from each proof item, prioritizing solved lemmas over
        formalizations. Removes duplicate imports and combines all code blocks
        into a single, compilable Lean 4 file.
        
        Returns:
            Complete Lean 4 code with imports and all proof steps
            
        Raises:
            ValueError: If no proof items are available (run autoformalize_series first)
        """
        if self.proof_items is None:
            raise ValueError("No proof items found. Run autoformalize_series or autoformalize_concurrent first.")
        

        code_blocks = []
        
        for item in self.proof_items:

            # Skip if item is of type TheoremCondition (except if it has a conditions/solutions, in order words no "variable" in lean_code)
            if type(item).__name__ in ('TheoremCondition', 'Definition'):
                if hasattr(item, 'formalization'):
                    if "variable" in item.formalization.get('lean_code', ""):
                        continue
                else:
                    continue
                
            code = None
            # First check solved_lemma["lean_code"]
            if hasattr(item, 'solved_lemma') and item.solved_lemma not in (None, {}):
                if item.solved_lemma["lean_verify"]:
                    code = item.solved_lemma.get('lean_code')
            
            # If not found, check formalization["lean_code"]
            if not code and hasattr(item, 'formalization') and item.formalization not in (None, {}):
                code = item.formalization.get('lean_code')        
            if code:
                # Remove unwanted lines
                cleaned_code = remove_imports(code)
                if cleaned_code:
                    code_blocks.append(cleaned_code)
        
        # Add standard imports at the beginning
        header = """import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat Filter"""
        
        if code_blocks:
            return header + "\n\n" + "\n\n".join(code_blocks)
        else:
            return header

    # ===== VISUALIZATION =====
    
    def plot_dag(self, filepath: str = "proof_dag.png") -> None:
        """
        Create a static visualization of the proof dependency graph.
        
        Generates a PNG image showing the proof structure with nodes representing
        proof steps and edges showing dependencies between them.
        
        Args:
            filepath: Path where to save the PNG image
            
        Raises:
            ValueError: If no proof items are available
        """
        if not self.proof_items:
            raise ValueError("No proof items found. Run autoformalize_series or autoformalize_concurrent first.")
        
        graph, node_info = build_dag([item.model_dump() for item in self.proof_items])
        create_static_visualization(graph, node_info, filepath)
        
    def interactive_dag(self, filepath: str = "proof_dag.html") -> None:
        """
        Create an interactive HTML visualization of the proof dependency graph.
        
        Generates an interactive HTML file that allows exploring the proof structure
        with zoom, pan, and click interactions. Includes the original proof text.
        
        Args:
            filepath: Path where to save the HTML file
            
        Raises:
            ValueError: If no proof items are available
        """
        if not self.proof_items:
            raise ValueError("No proof items found. Run autoformalize_series or autoformalize_concurrent first.")
        
        graph, node_info = build_dag([item.model_dump() for item in self.proof_items])
        create_interactive_visualization(G = graph, node_info = node_info, proof_str = self.nl_proof, filename = filepath)

    # ===== ANALYSIS & METRICS =====
    
    def summary(self, verbose: bool = True, pass_at: int = 100000) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the proof formalization results.
        
        Calculates success rates for formalization and proof generation, along with
        performance metrics like token usage and processing time.
        
        Args:
            verbose: Whether to print summary information to console
            pass_at: Maximum number of attempts to consider for success calculation
            
        Returns:
            Dictionary containing:
                - form_total: Total number of formalizable steps
                - form_correct: Number of successfully formalized steps
                - solv_total: Total number of solvable steps
                - solv_correct: Number of successfully solved steps
                - form_acc: Formalization accuracy (0.0 to 1.0)
                - solv_acc: Proof success rate (0.0 to 1.0)
                - generated_tokens: Total tokens generated
                - total_calls: Total number of LLM calls made
                - total_time: Total processing time in seconds
        """
        # Print number of tc, l, ts
        if verbose:
            self._print_progress_summary(None)

        # Count formalization successes
        formalized_count = 0
        total_formalizable = 0
        formalizer_llm_calls = 0
        # Count solving successes
        solved_count = 0
        total_solvable = 0
        solver_llm_calls = 0

        for item in self.proof_items:
            # Check formalization status
            if hasattr(item, 'formalization') and item.formalization not in (None, {}):
                total_formalizable += 1
                formalizer_llm_calls += min(item.formalization["tries"], pass_at) if "tries" in item.formalization else 1
                if 'lean_pass' in item.formalization and item.formalization['lean_pass']:
                    if "tries" in item.formalization and item.formalization["tries"] <= pass_at:
                        formalized_count += 1

            
            # Check solving status
            if hasattr(item, 'solved_lemma') and item.solved_lemma not in (None, {}):
                total_solvable += 1
                solver_llm_calls += min(item.solved_lemma["tries"],  pass_at) if "tries" in item.solved_lemma else 1
                if 'lean_verify' in item.solved_lemma and item.solved_lemma['lean_verify']:
                    if item.solved_lemma["tries"] <= pass_at:
                        solved_count += 1

        if verbose:
            print(f"{formalized_count} steps successfully formalized out of {total_formalizable}")
            print(f"{solved_count} steps successfully solved out of {total_solvable}")

        #add gen_tokens, and total time, ignore scorer and negation prover
        new_tokens = 0
        new_llm_call_logs = []
        for call in self.llm_call_logs:
            if isinstance(call["generated_tokens"], int):
                if "**not to prove the given theorem/lemma" not in ''.join(d['content'] for d in call["messages"]):
                    new_tokens += call["generated_tokens"]
                    new_llm_call_logs.append(call)  # Append the call to the new list
        total_time = new_llm_call_logs[-1]['end_time']-new_llm_call_logs[0]['start_time']

        return {"form_total": total_formalizable,
                "form_correct": formalized_count,
                "solv_total": total_solvable,
                "solv_correct": solved_count,
                "form_acc": formalized_count/total_formalizable,
                "solv_acc": solved_count/total_solvable,
                "generated_tokens": new_tokens,
                "total_calls": 1 + formalizer_llm_calls + solver_llm_calls,
                "total_time": total_time}
    
    def elapsed_time(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze timing information for all LLM calls made during processing.
        
        Args:
            verbose: Whether to print timing breakdown to console
            
        Returns:
            List of dictionaries containing model name and elapsed time for each call
        """
        total_time = self.llm_call_logs[-1]['end_time']-self.llm_call_logs[0]['start_time']
        if verbose:
            print(f"Total time: {total_time:2f}")
            print()
        elapsed_times = []
        for current in self.llm_call_logs:
            model = current["model"]
            elapsed = current["end_time"] - current["start_time"]
            elapsed_times.append({"model": model, "elapsed_time":elapsed, "prove_negation": "**not to prove the given theorem/lemma" in ''.join(d['content'] for d in current["messages"])})
            if verbose:
                print(f"Model: {model:<30} | Time: {elapsed:.2f}")
        
        return elapsed_times
    
    def proof_score(self, verbose: bool = True) -> None:
        """
        Compute and aggregate a formal proof's total score based on individual step scores.

        The method first checks if each proof step has a pre-existing semantic score. If a step lacks a score, 
        it's evaluated using a language model (`run_node_scorer_prompt`). The score and feedback are then 
        stored in the `item.score` attribute.

        After scoring all steps, it constructs a directed acyclic graph (DAG) of the proof. The total score
        is then computed by aggregating the individual step scores based on the specified `aggregation` method.
        The final score is stored in `self.total_score` and printed.

        Args:
            aggregation: The method to aggregate scores. Options are 'katz' (Katz centrality),
                        'laplacian' (Laplacian centrality), or 'equal' (equal weights).
            verbose: Whether to print scoring progress to console
        """
                
        if not self.score_model_manager:
            raise TypeError("Please provide a valid score_model_manager when you initiatilize AutoFormalizer.")
        
        total = len(self.proof_items)
        for idx,item in enumerate(self.proof_items):
            if not (hasattr(item, 'score') and item.score):
                if verbose:
                    self._print_status(f"\u27A4 Checking score of step {idx+1}/{total}: {item.id} ...", style='okblue')
                semantic_score, feedback = run_node_scorer_prompt(item, model_manager=self.score_model_manager, logs=self.llm_call_logs)
                item.score = {'semantic_score' : semantic_score, 'semantic_feedback':  feedback}
            if verbose:
                self._print_status(f"   \u2714 Score of step {idx+1}/{total}: {item.id} is {item.score['semantic_score']}", style='okgreen')

    
    def total_score(self, pass_at: int = 100000, aggregation = "equal"):
        """
        Computes the total score, penalizing items that exceed the maximum number of tries.
        
        Args:
            max_tries (float): The maximum number of attempts allowed for a formalization step.
                            Defaults to infinity, meaning no limit.
        """
        graph, node_info = build_dag([item.model_dump() for item in self.proof_items])
        new_proof_items = deepcopy(self.proof_items)
        
        for item in new_proof_items:
            if hasattr(item, 'formalization') and hasattr(item, 'score'):
                if "tries" in item.formalization:
                    if item.formalization["tries"] > pass_at:
                        item.score["semantic_score"] = 0
            else:
                print(f"Warning: Item {item} is missing 'formalization' or 'score' attributes. Returning None")
                return None

        return compute_total_score(new_proof_items, graph, aggregation=aggregation)
    
    
    def error_analysis(
        self,
        score_threshold: float = 0.6,
        prover_retries: int = 3,
        verbose: bool = True,
    ) -> None:
        """
        Perform comprehensive error analysis on the formalized proof.
        
        Analyzes each proof step to identify potential issues with formalization,
        proof generation, or semantic consistency. Categorizes errors into types
        like "Formalization", "Prover", "NL statement", etc.
        
        Args:
            score_threshold: Minimum semantic score to consider a formalization correct
            prover_retries: Number of retries for negation proof attempts
            verbose: Whether to print error analysis progress to console
        """

        total = len(self.proof_items)
        for idx, item in enumerate(self.proof_items):
            # Default
            error_type = None
            error_report = ""

            score = getattr(item, "score", {}) or {}
            solved_lemma = getattr(item, "solved_lemma", {}) or {}
            formalization = getattr(item, "formalization", {}) or {}

            if verbose:
                self._print_status(f"➤ Checking errors in step {idx+1}/{total}: {getattr(item, 'id', idx)} …", style='okblue')

            # 0) Formalized lean code not valid
            if not formalization.get("lean_pass", False):
                error_type = "Formalizer"
                error_report = "Formalized statement is not a valid Lean4 statement"
                item.error_report = {"error_type": error_type, "error_report": error_report}
                if verbose:
                    self._print_status(f"   → {error_type}: {error_report}", style="warning")
                continue

            # 1) Require semantic_score; if missing, record and continue.
            sem_score = score.get("semantic_score", None)
            if sem_score is None:
                error_type = "MissingScore"
                error_report = "Skipped: no `semantic_score` found in item.score; cannot assess formalization."
                # keep error_report minimal & primitive
                item.error_report = {"error_type": error_type, "error_report": error_report}
                if verbose:
                    self._print_status(f"   → {error_type}: {error_report}", style="warning")
                continue

            # 2) High semantic score path
            if isinstance(sem_score, (int, float)) and sem_score > score_threshold:
                if isinstance(item, TheoremCondition) or isinstance(item, Definition): #If item is theorem condition or definition, skip
                    error_type = None
                    error_report = "Natural-language lemma and formalization appear consistent (lean_verify = True)."
                elif bool(solved_lemma.get("lean_verify")):
                    error_type = None
                    error_report = "Natural-language lemma and formalization appear consistent (lean_verify = True)."
                else:
                    # Run prover for NEGATION, but store only primitive summary on item
                    results = run_solver_prompt(
                        item,
                        lean_server=self.lean_server,
                        model_manager=self.solver_model_manager,
                        logs=self.llm_call_logs,
                        max_retries=prover_retries,
                        prove_negation=True
                    )

                    # Store a trimmed, cycle-safe snapshot
                    if isinstance(results, dict):
                        item.solved_negation = {
                            "lean_verify": bool(results.get("lean_verify", False)),
                            "lean_code": results.get("lean_code") if isinstance(results.get("lean_code"), str) else None,
                        }
                        neg_ok = item.solved_negation["lean_verify"]
                    else:
                        item.solved_negation = {"lean_verify": False, "lean_code": None}
                        neg_ok = False

                    if neg_ok:
                        error_type = "NL statement"
                        error_report = (
                            f"Formalization appears correct (score={sem_score} > {score_threshold}), "
                            "and the prover could prove the NEGATION of the NL lemma. The NL statement is likely incorrect."
                        )
                    else:
                        error_type = "Prover"
                        error_report = (
                            f"Formalization appears correct (score={sem_score} > {score_threshold}), "
                            "but the prover could neither prove the lemma nor its negation. "
                            "This may be due to prover limitations, missing lemmas, or incomplete context."
                        )

            # 3) Low semantic score path
            else:
                feedback = score.get("semantic_feedback", "No feedback available.")
                parts = [
                    f"Formalization is likely incorrect: semantic score is {sem_score}, below threshold {score_threshold}.",
                    f"Scorer feedback: {feedback}",
                ]
                error_report = "\n".join(parts)
                error_type = "Formalizer"

            # Final minimal assignment (cycle-safe)
            item.error_report = {
                "error_type": error_type if error_type is None or isinstance(error_type, str) else str(error_type),
                "error_report": error_report if isinstance(error_report, str) else str(error_report),
            }

            if verbose:
                self._print_status(f"   → {item.error_report['error_type']}: {item.error_report['error_report']}", style="warning")

    # ===== I/O OPERATIONS =====
    
    def save(self, filepath: str, format: str = 'pickle') -> None:
        """
        Save the ProofFlow instance data to a file.
        
        Saves essential fields including proof_items, nl_proof, and llm_call_logs
        to allow for later restoration of the formalization state.
        
        Args:
            filepath: Path where to save the data
            format: File format - 'pickle' or 'json' (default: 'pickle')
                   Note: 'json' requires proof_items to be serializable
        
        Examples:
            >>> proof_flow.save("my_proof.pkl")
            >>> proof_flow.save("my_proof.json", format="json")
        """
        from .io import save_proofflow
        save_proofflow(self, filepath, format)
    
    def load(self, filepath: str, format: str = 'pickle') -> None:
        """
        Load ProofFlow instance data from a file.
        
        Restores essential fields including proof_items, nl_proof, and llm_call_logs
        from a previously saved file.
        
        Args:
            filepath: Path from where to load the data
            format: File format - 'pickle' or 'json' (default: 'pickle')
        
        Examples:
            >>> proof_flow.load("my_proof.pkl")
            >>> proof_flow.load("my_proof.json", format="json")
        
        Note:
            This method will overwrite the current proof_items, nl_proof, 
            and llm_call_logs with the loaded data.
        """
        from .io import load_proofflow
        load_proofflow(self, filepath, format)

