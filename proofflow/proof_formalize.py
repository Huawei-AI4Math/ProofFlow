import re

import httpx

from .lean_check import LeanServer, process_lean_string
from .utils import LLMManager, remove_imports


def extract_code_validate(text_input, lean_server):
    """Extracts the last Lean 4 code block from the model's output."""
    try:
        matches = re.findall(r"```lean4\n(.*?)\n```", text_input, re.DOTALL)
        if not matches:
            raise ValueError("No Lean 4 code block found.")
    except RuntimeError as e:
        return f"Error during code extraction: {str(e)}. Is ```lean4 ``` written?"

    response = matches[-1].strip()
    response = process_lean_string(response)  # add missing imports

    try:
        lean_pass, lean_verify, error_msg = lean_server.check_lean_string(response)
    except RuntimeError as e:
        raise ValueError("Error during Lean code verification " + str(e))

    return {
        "lean_code": response,
        "lean_pass": lean_pass,
        "error_msg": [] if lean_pass else error_msg,
    }


def run_formalizer_prompt(
    item: dict,
    lean_server: LeanServer,
    model_manager: LLMManager,
    all_items: list = None,
    logs=None,
    max_retries: int = 3,
    previous_context: bool = True,  # when formalizing step i provide formalized code of dependencies
    original_proof: str = "",  # when formalizing step i provide original proof
) -> tuple:
    """
    Builds the prompt string for the item and calls the LLM API.
    Uses the .md file as a system prompt.
    If the item has dependencies, appends their filtered dicts to the prompt string.
    Returns the parsed and validated JSON, with retry logic if validation fails.
    """

    is_condition_or_def = item.id.startswith("tc_") or item.id.startswith(
        "def_"
    )  # is it theorem condition or not?
    lemma_header = f"lemma {item.id}"
    dependencies = item.dependencies

    if not is_condition_or_def:
        # Combine all parts
        user_prompt_content = f"""Please autoformalize the following natural language problem proof step in Lean 4.
Use the following lemma name: {lemma_header}
The natural language statement is: {item.statement}
The dependencies are: {dependencies}

This is the  lean code skeleton you need to use:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

{lemma_header}
[place correct hypothesis here] :
[place goal here] := by
sorry
```

Important: **Please write only one lemma or theorem**!!
"""
    else:
        user_prompt_content = rf"""Please autoformalize the following natural language theorem condition in Lean 4.
Use the following name: {item.id}

The natural language statement is: {item.statement}

These the lean code skeleton you need to use (please make needed changes and fill ????):

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

variable [place correct hypothesis here]
```

Do not produce a theorem or a proof. Only provide the Lean 4 code.
Warning: this is not a lemma/theorem, it is a theorem condition. For this problem make use of "variable" and follow the following examples.

Name: tc\_1; Statement: Let \$(a\_n)\$ be a sequence of positive real numbers.
Lean 4 formalization:

```lean4
variable (a : ℕ → ℝ)
(tc_1 : ∀ n, 0 < a n)
```

Name: tc\_2; Statement: Let \$A\$ be a \$2 × 2\$ real matrix with eigenvalues \$\lambda\_1 = 3\$ and \$\lambda\_2 = -2\$.
Lean 4 formalization:

```lean4
variable (A : Matrix (Fin 2) (Fin 2) ℝ)
(tc_2 : ∃ v1 v2 : Fin 2 → ℝ, v1 ≠ 0 ∧ v2 ≠ 0 ∧ A.vecMul v1 = 3 • v1 ∧ A.vecMul v2 = -2 • v2)
```"""
    # Context can be: previous lean4 code or just goal statements and/or NL statement and/or original proof
    if previous_context:
        previous_context_str = [
            f"\n\n This proof step depend on previous proof steps, namely steps {dependencies}."
        ]
        previous_context_str.append(
            "Please make use use of their formal lean4 code, which contains relevant lean4 hypothesis and type declarations you may use:"
        )
        for d in all_items:
            if d.id in dependencies:
                previous_context_str.append("/n")  # Step {d.id}:")
                if (
                    hasattr(d, "formalization")
                    and d.formalization
                    and d.formalization["lean_code"]
                    and d.formalization["lean_pass"]
                ):  # check if lean code exists and it runs!
                    previous_context_str.append(
                        remove_imports(d.formalization["lean_code"])
                    )
                else:
                    previous_context_str.append(
                        f"Lean code not found or incorrect. Here is the natural language statement of step {d.id}: {d.statement}"
                    )
        previous_context_str.append(
            "/n Focus on the original formalization task I gave you and use the prebious Lean codes, extra context, type declarations, variables domains, etc. You can assume the information is correct. Make use of it!"
        )
        previous_context_str = "/n".join(previous_context_str)
        user_prompt_content += previous_context_str

    if original_proof:
        user_prompt_content += "\n\n This formalization task is a proof step which is part of a larger full proof given next:\n"
        user_prompt_content += original_proof
        user_prompt_content += "\nThe full proof may contain extra missing information that you need, specially variable types and domains (e.g 'r' is real and positive). Make use of it, specially if you encounter errors."
        user_prompt_content += "\nHowever, please focus on the original formalization task I gave you and use the previous full proof for extra context only."

    messages = [{"role": "user", "content": user_prompt_content}]

    formalization = {
        "lean_code": "",
        "lean_pass": False,
        "error_msg": "Failure to get a valid formalization from the LLM",
    }

    attempt_history = []

    for attempt in range(max_retries):
        try:
            response, messages = model_manager.call_llm(messages, logs=logs)
        except httpx.TimeoutException as e:
            print("OpenAI request failed:", e)

        # Try to validate the response
        try:
            formalization = extract_code_validate(response, lean_server)
            formalization["tries"] = attempt + 1
            formalization["attempt_history"] = attempt_history

            # check if lean is correct -> if yes end loop here
            if formalization["lean_pass"]:
                return formalization
            else:  # if not ajust prompt_str
                messages.append(
                    {
                        "role": "user",
                        "content": f"Lean error: "
                        + str(formalization["error_msg"])
                        + "\n\nBased on the error, please correct the previous response. ",
                    }
                )

        except ValueError as e:
            messages.append(
                {
                    "role": "user",
                    "content": f"\n\nError: "
                    + str(e)
                    + "\n\nBased on the error, please correct the previous response. ",
                }
            )

        attempt_history.append(formalization)

    return formalization
