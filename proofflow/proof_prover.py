import re

import httpx

from .lean_check import LeanServer, process_lean_string
from .proof_graph import Definition, TheoremCondition
from .utils import LLMManager


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
    except Exception as e:
        raise RuntimeError("Error during Lean code verification " + str(e))

    return {
        "lean_code": response,
        "lean_pass": lean_pass,
        "lean_verify": lean_verify,
        "error_msg": error_msg,
    }  # None if lean_verify else error_msg


def run_solver_prompt(
    item: dict,
    lean_server: LeanServer,
    model_manager: LLMManager,
    logs=None,
    max_retries: int = 3,
    prove_negation: bool = False,
) -> tuple:
    """
    Builds the prompt string for the item and calls the LLM API.
    Uses the .md file as a system prompt.
    If the item has dependencies, appends their filtered dicts to the prompt string.
    Returns the parsed and validated JSON, with retry logic if validation fails.
    """

    # Deal with non-correct input cases
    is_condition_or_def = isinstance(item, TheoremCondition) or isinstance(
        item, Definition
    )
    if is_condition_or_def:
        return {}
    if not item.formalization["lean_code"]:
        return {}

    if prove_negation:
        user_prompt_content = f"""
Your task is **not to prove the given theorem/lemma, but to disprove it by proving its logical negation**.

This is the original lemma/theorem statement I want you to refute:
{item.statement}

Below is the Lean 4 code for the statement. 
You must instead negate the goal and then attempt to prove that negation in Lean 4. 
If the statement cannot be directly negated syntactically, carefully construct the logically equivalent negation.

Please output only valid Lean 4 code with the negated theorem and a proof attempt.

```lean4
{item.formalization["lean_code"]}
```"""

    else:
        user_prompt_content = f"""
This is the lemma/theorem I want you to prove:
{item.statement}

Complete the following Lean 4 code (**do not remove imports**):

```lean4
{item.formalization["lean_code"]}
```

You can adapt previous lean4 lemma statement to fit the goal, specially if you encounter errors.
"""
    if not item.formalization["lean_pass"]:
        user_prompt_content += "/n The previous Lean4 code I sent you contains errors. Please take that into account."

    messages = [{"role": "user", "content": user_prompt_content}]

    results = {
        "lean_code": "",
        "lean_pass": False,
        "lean_verify": False,
        "error_msg": "failure to get any response from LLM",
    }

    attempt_history = []
    for attempt in range(max_retries):
        try:
            response, messages = model_manager.call_llm(messages, logs=logs)
        except httpx.TimeoutException as e:
            print("OpenAI request failed:", e)

        # Try to validate the response
        try:
            results = extract_code_validate(response, lean_server)
            results["tries"] = attempt + 1
            results["attempt_history"] = attempt_history

            # check if lean is correct -> if yes end loop here
            if results["lean_verify"]:
                return results
            else:  # if not ajust prompt_str
                messages.append(
                    {
                        "role": "user",
                        "content": f"Lean error/warnings: "
                        + str(results["error_msg"])
                        + "\n\n Based on these errors, please correct the previous response. ",
                    }
                )
        except ValueError as e:
            messages.append(
                {
                    "role": "user",
                    "content": f"\n\nError: "
                    + str(e)
                    + "\n\n Based on these errors, please correct the previous response. ",
                }
            )

        attempt_history.append(results)

    return results
