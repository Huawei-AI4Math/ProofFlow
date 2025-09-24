from proofflow import LeanServer
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import time
import re
import os
import json
import uuid
import argparse
import random

load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("OPEN_AI_BASE_URL")
LEANSERVER_URL = os.getenv("LEAN_SERVER_URL")

# Thinking mode is enabled
MODEL_NAME="google/gemini-2.5-pro"
THINKING_MODE = True


# Lean Server
lean_server = LeanServer(api_url=LEANSERVER_URL) 


def api_call_with_retry(user_prompt: str, max_retries: int = 3) -> Tuple[str, int, float]:
    
    if not THINKING_MODE:
        extra_body_defined = {
                    "reasoning": {
                    "max_tokens": 0  # set to 0 to disable thinking
                    }
                }
    else:
        extra_body_defined={
                    "reasoning": {
                    "effort": "high"  # optional: "low", "medium", "high"
                    }
                }         

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    for attempt in range(max_retries):
        try:
            # record start time
            start_time = time.time()

            completion = client.chat.completions.create(
                extra_body=extra_body_defined,
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ]
            )

            # record end time
            end_time = time.time()
            call_time = end_time - start_time

            # extract response content
            response_content = completion.choices[0].message.content

            # get token usage
            output_tokens = 0
            if hasattr(completion, 'usage') and completion.usage:
                output_tokens = completion.usage.completion_tokens if hasattr(completion.usage,
                                                                              'completion_tokens') else 0

            return response_content, output_tokens, call_time

        except Exception as e:
            print(f"Failure in attempt {attempt + 1}: {str(e)}")

            if attempt == max_retries - 1:
                return f"API call failed: {str(e)}", 0, 0.0

            # wait before retrying
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Waiting {wait_time:.1f} seconds before retry...")
            time.sleep(wait_time)

    return "API call failed: All retries failed", 0, 0.0

# New prompts for step-by-step formalization
PROBLEM_TRANSLATION_PROMPT = """
You are a Lean 4 theorem proving specialist with expertise in formal mathematics and proof assistant systems. Your task is to translate an informal mathematical theorem statement into a rigorous, syntactically correct Lean 4 formal theorem statement.

## Instructions:
1. Carefully analyze the informal theorem statement to identify all mathematical objects, hypotheses, and the main goal.
2. Choose appropriate Lean 4 types and notation for all mathematical concepts.
3. Ensure the formal statement captures the complete mathematical meaning of the informal statement.
4. Use proper Lean 4 syntax, end the theorem with ":= by".
5. Keep the header of the Lean 4 code unchanged.

## Current Translation Task:

### Informal Theorem Statement:
{theorem}

### Formal Lean 4 Theorem Statement:
```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

(output the Lean 4 theorem statement here)
```

Output only the Lean 4 theorem statement, wrapped between "```lean4" and "```".
"""

PROBLEM_TRANSLATION_REFINE_PROMPT = """
You are a Lean 4 theorem proving specialist with expertise in formal mathematics and proof assistant systems. Your task is to translate an informal mathematical theorem statement into a rigorous, syntactically correct Lean 4 formal theorem statement.

## Instructions:
1. Carefully analyze the informal theorem statement to identify all mathematical objects, hypotheses, and the main goal.
2. Choose appropriate Lean 4 types and notation for all mathematical concepts.
3. Ensure the formal statement captures the complete mathematical meaning of the informal statement.
4. Use proper Lean 4 syntax, end the theorem with ":= by".
5. Keep the header of the Lean 4 code unchanged.

## Current Translation Task:

### Informal Theorem Statement:
{theorem}

### Previous Formal Lean 4 Theorem Statement:
```lean4
{last_lean_code}
```

The previous Lean4 code I sent you contains errors. Please take that into account.
Lean error/warnings: {error_massage}

Based on these errors, please correct the previous response. 

### Formal Lean 4 Theorem Statement:
```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

(output the Lean 4 theorem statement here)
```

Output only the Lean 4 theorem statement, wrapped between "```lean4" and "```".
"""

STEP_TRANSLATION_PROMPT = """
You are a Lean 4 theorem proving specialist with expertise in formal mathematics and proof assistant systems. Your task is to translate a single natural language proof step into the corresponding Lean 4 tactic(s), given the current proof context.

## Instructions:
1. Examine the previous Lean 4 proof steps to understand the current proof state, available hypotheses, and the goal to be proven.
2. Carefully analyze the current proof step described in natural language, identifying the mathematical reasoning, logical operations, and transformations involved.
3. Apply appropriate Lean 4 tactics to ensure syntax correctness, pay great attention on the INDENT, all proofs should be indented one space more than theorems.
4. Maintain mathematical rigor while preserving the original proof's essence.
5. DO NOT USE "SORRY" IN YOUR TACTIC OUTPUT.

## Current Translation Task:

### Previous Lean 4 Proof Steps:
```lean4
{previous_steps}
```

### Current Natural Language Proof Step:
{current_step}

### Required Lean 4 Tactic(s):
```lean4
(output the Lean 4 tactic(s) here)
```

Output only the Lean 4 tactic(s) needed for this step, wrapped between "```lean4" and "```". Do not include the full theorem statement or previous steps in your output. Give answer straight away in lean code block.
"""

STEP_TRANSLATION_REFINE_PROMPT = """
You are a Lean 4 theorem proving specialist with expertise in formal mathematics and proof assistant systems. Your task is to translate a single natural language proof step into the corresponding Lean 4 tactic(s), given the current proof context.

## Instructions:
1. Examine the previous Lean 4 proof steps to understand the current proof state, available hypotheses, and the goal to be proven.
2. Carefully analyze the current proof step described in natural language, identifying the mathematical reasoning, logical operations, and transformations involved.
3. Apply appropriate Lean 4 tactics to ensure syntax correctness, pay great attention on the INDENT, all proofs should be indented one space more than theorems.
4. Maintain mathematical rigor while preserving the original proof's essence.
5. DO NOT USE "SORRY" IN YOUR TACTIC OUTPUT.

## Current Translation Task:

### Previous Lean 4 Proof Steps:
```lean4
{previous_steps}
```

### Current Natural Language Proof Step:
{current_step}

### Previous Required Lean 4 Tactic(s):
```lean4
{last_lean_code}
```

The previous Lean4 code I sent you contains errors. Please take that into account.
Lean error/warnings: {error_massage}

Based on these errors, please correct the previous response. Output only the Lean 4 tactic(s) needed for this step, wrapped between "```lean4" and "```". Do not include the full theorem statement or previous steps in your output. Give answer straight away in lean code block.
"""


def theorem_translation(theorem: str, self_refine: bool = False, last_lean_code: str = "", error_massage: str = "") -> Tuple[str, int, float]:
    """
    Process theorem translation
    Returns: (response content, output token count, call time)
    """
    if self_refine:
        prompt = PROBLEM_TRANSLATION_REFINE_PROMPT.format(
            theorem=theorem.strip(),
            last_lean_code=last_lean_code.strip(),
            error_massage=error_massage.strip()
        )
    else:
        prompt = PROBLEM_TRANSLATION_PROMPT.format(theorem=theorem.strip())
    return api_call_with_retry(prompt)

def step_by_step_formalization(previous_steps: str, current_step: str, self_refine: bool = False, last_lean_code: str = "", error_massage: str = "") -> Tuple[str, int, float]:
    """
    Process individual proof step formalization
    Returns: (response content, output token count, call time)
    """
    if self_refine:
        prompt = STEP_TRANSLATION_REFINE_PROMPT.format(
            previous_steps=previous_steps.strip(),
            current_step=current_step.strip(),
            last_lean_code=last_lean_code.strip(),
            error_massage=error_massage.strip()
        )
    else:
        prompt = STEP_TRANSLATION_PROMPT.format(
            previous_steps=previous_steps.strip(),
            current_step=current_step.strip()
        )
    return api_call_with_retry(prompt)


def extract_code_validate(text_input, lean_server):
    """Extracts the last Lean 4 code block from the model's output."""
    try:
        matches = re.findall(r'```lean4\n(.*?)\n```', text_input, re.DOTALL)
        if not matches:
            raise ValueError("No Lean 4 code block found.")
    except RuntimeError as e:
        return f"Error during code extraction: {str(e)}. Is ```lean4 ``` written?"
    
    response = matches[-1].strip() 

    try:
        lean_pass, lean_verify, error_msg = lean_server.check_lean_string(response)
        if not isinstance(error_msg, str):
            error_msg = str(error_msg)
    except Exception as e:
        raise RuntimeError("Error during Lean code verification " + str(e))
    
    return {"lean_code": response, "lean_pass": lean_pass, "lean_verify": lean_verify, "error_msg": error_msg} #None if lean_verify else error_msg

def add_sorry_with_indent(text):
    lines = text.split('\n')
    
    if not lines:
        return "sorry"

    indent = ' ' * 2
    sorry_line = indent + 'sorry'
    
    if text.endswith('\n'):
        return text + sorry_line
    else:
        return text + '\n' + sorry_line


def llm_and_verify_with_retry_theorem(theorem: str, lean_server, max_retries: int, selfrefine_mode: bool = False) -> Tuple[str, Dict[str, Any], int, float]:
    """
    Complete retry workflow for theorem LLM generation + Lean verification
    When verification fails, it will recall LLM to generate new response and verify again
    
    Args:
        theorem: Theorem content
        lean_server: Lean server instance
        max_retries: Maximum verification retry attempts
        selfrefine_mode: Whether to enable self-refine mode, when enabled will feedback error info to LLM
    
    Returns:
        tuple: (final LLM response, Lean verification results, total tokens, total time)
    """
    attempt_history = []
    total_tokens = 0
    total_time = 0.0
    final_llm_response = ""
    last_lean_code = ""
    last_error_msg = ""
    
    for retry_attempt in range(max_retries):
        print(f"  Theorem retry {retry_attempt + 1}/{max_retries}: LLM generation + Lean verification...")
        
        # Step 1: Call LLM to generate response
        # For first attempt or non-self-refine mode, use original calling method
        if retry_attempt == 0 or not selfrefine_mode:
            llm_response, output_tokens, call_time = theorem_translation(theorem)
        else:
            # Self-refine mode: pass previous code and error information
            llm_response, output_tokens, call_time = theorem_translation(
                theorem,
                self_refine=True, 
                last_lean_code=last_lean_code,
                error_massage=last_error_msg
            )

        total_tokens += output_tokens
        total_time += call_time
        final_llm_response = llm_response
        
        # If LLM API call fails, return directly
        if "API call failed" in llm_response:
            print(f"  LLM API call failed: {llm_response}")
            lean_results = {
                "lean_code": "",
                "lean_pass": False, 
                "lean_verify": False, 
                "error_msg": "LLM API call failed",
                "verified_code": "",
                "tries": retry_attempt + 1,
                "attempt_history": attempt_history,
                "total_tokens": total_tokens,
                "total_time": total_time,
                "avg_tokens_per_trial": total_tokens / (retry_attempt + 1) if retry_attempt >= 0 else 0,
                "avg_time_per_trial": total_time / (retry_attempt + 1) if retry_attempt >= 0 else 0
            }
            return final_llm_response, lean_results, total_tokens, total_time
        
        print(f"  LLM generation finished, tokens: {output_tokens}, time: {call_time:.2f}s")
        if selfrefine_mode and retry_attempt > 0:
            print(f"  Using self-refine mode, improving based on previous error")
        
        # Step 2: Extract theorem Lean code from response
        try:
            theorem_matches = re.findall(r'```lean4\n(.*?)```', llm_response, re.DOTALL)
            if not theorem_matches:
                raise ValueError("No theorem Lean code block found in response")
            # Only remove leading/trailing newlines, preserve internal indentation
            theorem_lean_code = theorem_matches[-1].strip('\n')
        except Exception as e:
            print(f"  Failed to extract theorem Lean code from response: {e}")
            # save last_lean_code for self-refine
            last_lean_code = ""
            last_error_msg = f"Code extraction failed: {e}"
            
            # If this is the last attempt, return failure result
            if retry_attempt == max_retries - 1:
                lean_results = {
                    "lean_code": "",
                    "lean_pass": False,
                    "lean_verify": False,
                    "error_msg": f"Code extraction failed: {e}",
                    "verified_code": "",
                    "tries": retry_attempt + 1,
                    "attempt_history": attempt_history,
                    "total_tokens": total_tokens,
                    "total_time": total_time,
                    "avg_tokens_per_trial": total_tokens / (retry_attempt + 1),
                    "avg_time_per_trial": total_time / (retry_attempt + 1)
                }
                return final_llm_response, lean_results, total_tokens, total_time
            continue
        
        # Step 3: Perform Lean verification on theorem
        try:
            # print(f"  Starting Lean verification for theorem...")
            # Build complete code for verification
            temp_complete_code = theorem_lean_code
            
            standard_header = '''import Mathlib
import Aesop

set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat Filter
'''
            # Add standard header if temp_complete_code starts with "theorem"
            if temp_complete_code.strip().startswith('theorem'):
                temp_complete_code = standard_header + temp_complete_code 
                
            temp_complete_code = temp_complete_code + (" sorry" if not temp_complete_code.endswith("sorry") else "")
            
            lean_results = extract_code_validate(f"```lean4\n{temp_complete_code}\n```", lean_server)
            is_pass = lean_results.get("lean_pass", False)
            
            lean_results["verified_code"] = temp_complete_code

            # Record this attempt
            attempt_record = {
                "attempt": retry_attempt + 1,
                "llm_response": llm_response,
                "theorem_lean_code": theorem_lean_code,
                "lean_pass": is_pass,
                "error_msg": lean_results.get("error_msg", ""),
                "verified_code": temp_complete_code,
                "tokens": output_tokens,
                "time": call_time,
                "used_selfrefine": selfrefine_mode and retry_attempt > 0
            }
            attempt_history.append(attempt_record)
            
            # If verification succeeds, return results
            if is_pass:
                print(f"  Lean verification successful!")
                lean_results["theorem_lean_code"] = theorem_lean_code
                lean_results["tries"] = retry_attempt + 1
                lean_results["attempt_history"] = attempt_history
                lean_results["total_tokens"] = total_tokens
                lean_results["total_time"] = total_time
                lean_results["avg_tokens_per_trial"] = total_tokens / (retry_attempt + 1)
                lean_results["avg_time_per_trial"] = total_time / (retry_attempt + 1)
                return final_llm_response, lean_results, total_tokens, total_time
            else:
                # Save error info and code for next self-refine use
                last_error_msg = lean_results.get("error_msg", "Unknown error")
                last_lean_code = theorem_lean_code
                print(f"  Lean verification failed: {last_error_msg}")
                
                # if this is the last attempt, return result (including last code)
                if retry_attempt == max_retries - 1:
                    print(f"  Final attempt failed, but keeping last generated code for continuation")
                    lean_results["theorem_lean_code"] = theorem_lean_code
                    lean_results["tries"] = retry_attempt + 1
                    lean_results["attempt_history"] = attempt_history
                    lean_results["total_tokens"] = total_tokens
                    lean_results["total_time"] = total_time
                    lean_results["avg_tokens_per_trial"] = total_tokens / (retry_attempt + 1)
                    lean_results["avg_time_per_trial"] = total_time / (retry_attempt + 1)
                    return final_llm_response, lean_results, total_tokens, total_time
                
                if selfrefine_mode and retry_attempt < max_retries - 1:
                    print(f"  Will use self-refine mode for improvement in next attempt")
                
        except Exception as e:
            error_msg = f"Verification exception: {str(e)}"
            print(f"  Lean verification exception: {e}")
            
            # Save exception info for self-refine use
            last_error_msg = error_msg
            last_lean_code = theorem_lean_code if 'theorem_lean_code' in locals() else ""
            
            # add verified_code for self-refine
            verified_code = temp_complete_code if 'temp_complete_code' in locals() else ""
            
            attempt_record = {
                "attempt": retry_attempt + 1,
                "llm_response": llm_response,
                "theorem_lean_code": last_lean_code,
                "lean_pass": False,
                "error_msg": error_msg,
                "verified_code": verified_code,
                "tokens": output_tokens,
                "time": call_time,
                "used_selfrefine": selfrefine_mode and retry_attempt > 0
            }
            attempt_history.append(attempt_record)
            
            # if this is the last attempt, return result (including last code)
            if retry_attempt == max_retries - 1:
                print(f"  Final attempt failed with exception, but keeping last generated code for continuation")
                lean_results = {
                    "theorem_lean_code": last_lean_code,
                    "lean_pass": False, 
                    "lean_verify": False, 
                    "error_msg": error_msg,
                    "verified_code": verified_code,
                    "tries": retry_attempt + 1,
                    "attempt_history": attempt_history,
                    "total_tokens": total_tokens,
                    "total_time": total_time,
                    "avg_tokens_per_trial": total_tokens / (retry_attempt + 1),
                    "avg_time_per_trial": total_time / (retry_attempt + 1)
                }
                return final_llm_response, lean_results, total_tokens, total_time
    
    # this branch is theoretically unreachable, as the above handles the last attempt
    print(f"  Final failure: attempted {max_retries} rounds of theorem generation + verification")
    last_verified_code = attempt_history[-1]["verified_code"] if attempt_history else ""
    lean_results = {
        "theorem_lean_code": last_lean_code,
        "lean_pass": False, 
        "lean_verify": False, 
        "error_msg": f"Theorem verification failed after {max_retries} retries",
        "verified_code": last_verified_code,
        "tries": max_retries,
        "attempt_history": attempt_history,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tokens_per_trial": total_tokens / max_retries if max_retries > 0 else 0,
        "avg_time_per_trial": total_time / max_retries if max_retries > 0 else 0
    }
    return final_llm_response, lean_results, total_tokens, total_time


def llm_and_verify_with_retry_onestep(previous_steps: str, current_step: str, lean_server, max_retries: int, selfrefine_mode: bool = False) -> Tuple[str, Dict[str, Any], int, float]:
    """
    Complete retry workflow for single step LLM generation + Lean verification
    When verification fails, it will recall LLM to generate new response and verify again
    
    Args:
        previous_steps: Previous Lean proof steps
        current_step: Current natural language proof step
        lean_server: Lean server instance
        max_retries: Maximum verification retry attempts
        selfrefine_mode: Whether to enable self-refine mode, when enabled will feedback error info to LLM
    
    Returns:
        tuple: (final LLM response, Lean verification results, total tokens, total time)
    """
    attempt_history = []
    total_tokens = 0
    total_time = 0.0
    final_llm_response = ""
    last_lean_code = ""
    last_error_msg = ""
    
    for retry_attempt in range(max_retries):
        print(f"    Step retry {retry_attempt + 1}/{max_retries}: LLM generation + Lean verification...")
        
        # Step 1: Call LLM to generate response
        # For first attempt or non-self-refine mode, use original calling method
        if retry_attempt == 0 or not selfrefine_mode:
            llm_response, output_tokens, call_time = step_by_step_formalization(previous_steps, current_step)
        else:
            # Self-refine mode: pass previous code and error information
            llm_response, output_tokens, call_time = step_by_step_formalization(
                previous_steps, current_step,
                self_refine=True, 
                last_lean_code=last_lean_code,
                error_massage=last_error_msg
            )

        total_tokens += output_tokens
        total_time += call_time
        final_llm_response = llm_response
        
        # If LLM API call fails, return directly
        if "API call failed" in llm_response:
            print(f"    LLM API call failed: {llm_response}")
            lean_results = {
                "lean_code": "",
                "lean_pass": False, 
                "lean_verify": False, 
                "error_msg": "LLM API call failed",
                "verified_code": "",
                "tries": retry_attempt + 1,
                "attempt_history": attempt_history,
                "total_tokens": total_tokens,
                "total_time": total_time,
                "avg_tokens_per_trial": total_tokens / (retry_attempt + 1) if retry_attempt >= 0 else 0,
                "avg_time_per_trial": total_time / (retry_attempt + 1) if retry_attempt >= 0 else 0
            }
            return final_llm_response, lean_results, total_tokens, total_time
        
        print(f"    LLM generation finished, tokens: {output_tokens}, time: {call_time:.2f}s")
        if selfrefine_mode and retry_attempt > 0:
            print(f"    Using self-refine mode, improving based on previous error")
        
        # Step 2: Extract Lean tactics from response
        try:
            step_matches = re.findall(r'```lean4\n(.*?)```', llm_response, re.DOTALL)
            if not step_matches:
                raise ValueError("No Lean code block found in step response")
            # Only remove leading/trailing newlines, preserve internal indentation
            step_lean_tactics = step_matches[-1].strip('\n')
        except Exception as e:
            print(f"    Failed to extract Lean code from step response: {e}")
            # save last_lean_code for self-refine
            last_lean_code = ""
            last_error_msg = f"Code extraction failed: {e}"
            
            # if this is the last attempt, return failure result
            if retry_attempt == max_retries - 1:
                lean_results = {
                    "lean_code": "",
                    "lean_pass": False,
                    "lean_verify": False,
                    "error_msg": f"Code extraction failed: {e}",
                    "verified_code": "",
                    "tries": retry_attempt + 1,
                    "attempt_history": attempt_history,
                    "total_tokens": total_tokens,
                    "total_time": total_time,
                    "avg_tokens_per_trial": total_tokens / (retry_attempt + 1),
                    "avg_time_per_trial": total_time / (retry_attempt + 1)
                }
                return final_llm_response, lean_results, total_tokens, total_time
            continue
        
        # Step 3: Perform Lean verification on current step
        try:
            # print(f"    Starting Lean verification for current step...")
            # Build temporary complete code for verification
            temp_complete_code = previous_steps + f"\n{step_lean_tactics}"
            
            standard_header = '''import Mathlib
import Aesop

set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat Filter
'''
            if temp_complete_code.strip().startswith('theorem'):
                temp_complete_code = standard_header + temp_complete_code 
                
            if not temp_complete_code.endswith("sorry"): 
                temp_complete_code = add_sorry_with_indent(temp_complete_code)
            
            lean_results = extract_code_validate(f"```lean4\n{temp_complete_code}\n```", lean_server)
            is_pass = lean_results.get("lean_pass", False)
            
            lean_results["verified_code"] = temp_complete_code

            # Record this attempt
            attempt_record = {
                "attempt": retry_attempt + 1,
                "llm_response": llm_response,
                "lean_tactics": step_lean_tactics,
                "lean_pass": is_pass,
                "error_msg": lean_results.get("error_msg", ""),
                "verified_code": temp_complete_code,
                "tokens": output_tokens,
                "time": call_time,
                "used_selfrefine": selfrefine_mode and retry_attempt > 0
            }
            attempt_history.append(attempt_record)
            
            # If verification succeeds, return results
            if is_pass:
                print(f"    Lean verification successful!")
                lean_results["lean_tactics"] = step_lean_tactics
                lean_results["tries"] = retry_attempt + 1
                lean_results["attempt_history"] = attempt_history
                lean_results["total_tokens"] = total_tokens
                lean_results["total_time"] = total_time
                lean_results["avg_tokens_per_trial"] = total_tokens / (retry_attempt + 1)
                lean_results["avg_time_per_trial"] = total_time / (retry_attempt + 1)
                return final_llm_response, lean_results, total_tokens, total_time
            else:
                # Save error info and code for next self-refine use
                last_error_msg = lean_results.get("error_msg", "Unknown error")
                last_lean_code = step_lean_tactics
                print(f"    Lean verification failed: {last_error_msg}")
                
                # if this is the last attempt, return result (including last code)
                if retry_attempt == max_retries - 1:
                    print(f"    Final attempt failed, but keeping last generated code for continuation")
                    lean_results["lean_tactics"] = step_lean_tactics
                    lean_results["tries"] = retry_attempt + 1
                    lean_results["attempt_history"] = attempt_history
                    lean_results["total_tokens"] = total_tokens
                    lean_results["total_time"] = total_time
                    lean_results["avg_tokens_per_trial"] = total_tokens / (retry_attempt + 1)
                    lean_results["avg_time_per_trial"] = total_time / (retry_attempt + 1)
                    return final_llm_response, lean_results, total_tokens, total_time
                
                if selfrefine_mode and retry_attempt < max_retries - 1:
                    print(f"    Will use self-refine mode for improvement in next attempt")
                
        except Exception as e:
            error_msg = f"Verification exception: {str(e)}"
            print(f"    Lean verification exception: {e}")
            
            # Save exception info for self-refine use
            last_error_msg = error_msg
            last_lean_code = step_lean_tactics if 'step_lean_tactics' in locals() else ""
            
            # add verified_code for self-refine
            verified_code = temp_complete_code if 'temp_complete_code' in locals() else ""
            
            attempt_record = {
                "attempt": retry_attempt + 1,
                "llm_response": llm_response,
                "lean_tactics": last_lean_code,
                "lean_pass": False,
                "error_msg": error_msg,
                "verified_code": verified_code,
                "tokens": output_tokens,
                "time": call_time,
                "used_selfrefine": selfrefine_mode and retry_attempt > 0
            }
            attempt_history.append(attempt_record)
            
            # if this is the last attempt, return result (including last code)
            if retry_attempt == max_retries - 1:
                print(f"    Final attempt failed with exception, but keeping last generated code for continuation")
                lean_results = {
                    "lean_tactics": last_lean_code,
                    "lean_pass": False, 
                    "lean_verify": False, 
                    "error_msg": error_msg,
                    "verified_code": verified_code,
                    "tries": retry_attempt + 1,
                    "attempt_history": attempt_history,
                    "total_tokens": total_tokens,
                    "total_time": total_time,
                    "avg_tokens_per_trial": total_tokens / (retry_attempt + 1),
                    "avg_time_per_trial": total_time / (retry_attempt + 1)
                }
                return final_llm_response, lean_results, total_tokens, total_time
    
    # this branch is theoretically unreachable, as the above handles the last attempt
    print(f"    Final failure: attempted {max_retries} rounds of step generation + verification")
    last_verified_code = attempt_history[-1]["verified_code"] if attempt_history else ""
    lean_results = {
        "lean_tactics": last_lean_code,
        "lean_pass": False, 
        "lean_verify": False, 
        "error_msg": f"Step verification failed after {max_retries} retries",
        "verified_code": last_verified_code,
        "tries": max_retries,
        "attempt_history": attempt_history,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tokens_per_trial": total_tokens / max_retries if max_retries > 0 else 0,
        "avg_time_per_trial": total_time / max_retries if max_retries > 0 else 0
    }
    return final_llm_response, lean_results, total_tokens, total_time



def process_single_theorem_step_by_step(theorem_data: Dict[str, Any], index: int, lean_server, max_retries: int = 3, max_retries_thm: int = 1, selfrefine_mode: bool = False) -> Dict[str, Any]:
    """
    Process single theorem with step-by-step formalization
    
    Args:
        theorem_data: Dictionary containing theorem information
        index: Theorem index
        lean_server: Lean server instance
        max_retries: Maximum retry attempts for each step and theorem
        max_retries_thm: Maximum retry attempts for entire theorem processing
        selfrefine_mode: Whether to enable self-refine mode
    
    Returns:
        dict: Processing result
    """
    print(f"Processing theorem {index + 1}...")
    
    # Extract theorem content
    theorem = theorem_data.get('nl_theorem', '')
    if not theorem:
        print(f"  Skipping theorem {index + 1} (missing theorem content)")
        return None
    
    # Try processing the entire theorem max_retries_thm times
    for thm_attempt in range(max_retries_thm):
        if thm_attempt > 0:
            print(f"  Theorem processing attempt {thm_attempt + 1}/{max_retries_thm}...")
        
        total_tokens = 0
        total_time = 0.0
        
        # Step 1: Translate theorem statement with retry and verification
        print(f"  Translating theorem statement...")
        theorem_response, theorem_lean_results, theorem_tokens, theorem_time = llm_and_verify_with_retry_theorem(
            theorem, lean_server, max_retries, selfrefine_mode
        )
        total_tokens += theorem_tokens
        total_time += theorem_time
        
        # Check if theorem translation succeeded
        if "API call failed" in theorem_response:
            print(f"  Theorem translation failed: {theorem_response}")
            if thm_attempt == max_retries_thm - 1:  # Last attempt
                return {
                    'id': theorem_data.get('id', f'theorem_{index + 1}'),
                    'original_theorem': theorem,
                    'theorem_translation': theorem_response,
                    'theorem_lean_results': theorem_lean_results,
                    'proof_steps': [],
                    'final_lean_code': "",
                    'lean_verification': {"lean_pass": False, "lean_verify": False, "error_msg": "Theorem translation failed due to API error.", "verified_code": ""},
                    'total_tokens': total_tokens,
                    'total_time': total_time
                }
            continue  # Try again
        
        # get theorem code and verification result
        theorem_lean_code = theorem_lean_results.get('theorem_lean_code', '')
        is_theorem_pass = theorem_lean_results.get('lean_pass', False)
        
        # if no theorem code is available, use empty template
        if not theorem_lean_code:
            print(f"  No theorem code available, using empty template")
            theorem_lean_code = "theorem placeholder : True := by sorry"
        
        # check if theorem verification failed
        verification_failed = False
        if not is_theorem_pass:
            print(f"  Theorem verification failed - marking all subsequent steps as failed")
            verification_failed = True
        else:
            print(f"  Theorem translation finished, tokens: {theorem_tokens}, time: {theorem_time:.2f}s")
        
        # Step 2: Read proof steps from theorem_data's proof_graph field
        try:
            steps_data = theorem_data.get('proof_graph', [])
            
            if not steps_data:
                print(f"  Warning: No proof_graph found in theorem_data")
                return {
                    'id': theorem_data.get('id', f'theorem_{index + 1}'),
                    'original_theorem': theorem,
                    'theorem_translation': theorem_response,
                    'theorem_lean_results': theorem_lean_results,
                    'proof_steps': [],
                    'final_lean_code': theorem_lean_code,
                    'lean_verification': {"lean_pass": False, "lean_verify": False, "error_msg": "No proof_graph found in theorem_data", "verified_code": ""},
                    'total_tokens': total_tokens,
                    'total_time': total_time
                }
                
        except Exception as e:
            print(f"  Failed to read proof_graph from theorem_data: {e}")
            return {
                'id': theorem_data.get('id', f'theorem_{index + 1}'),
                'original_theorem': theorem,
                'theorem_translation': theorem_response,
                'theorem_lean_results': theorem_lean_results,
                'proof_steps': [],
                'final_lean_code': theorem_lean_code,
                'lean_verification': {"lean_pass": False, "lean_verify": False, "error_msg": f"Failed to read proof_graph: {e}", "verified_code": ""},
                'total_tokens': total_tokens,
                'total_time': total_time
            }
        
        # Step 3: Translate proof steps step by step
        previous_steps = theorem_lean_code

        print(f"  Found {len(steps_data)} steps")
        print(f"  previous_steps: {previous_steps}")

        proof_steps = []
        
        # Filter steps that need proof (id doesn't contain "tc" or "ts")
        proof_items = [item for item in steps_data if not (item.get('id', '').startswith('tc') or item.get('id', '').startswith('def'))]
        
        print(f"  Found {len(proof_items)} proof steps")
        
        for step_idx, step_item in enumerate(proof_items):
            step_id = step_item.get('id', f'step_{step_idx}')
            natural_language_step = step_item.get('statement', '')
            
            if not natural_language_step:
                print(f"    Skip step {step_id} (missing natural language description)")
                continue
            
            print(f"    Processing proof step {step_id}...")
            
            # if verification failed, skip actual processing and mark as failure
            if verification_failed:
                print(f"    Step {step_id} skipped due to previous verification failure")
                placeholder_tactics = "sorry  -- skipped due to previous failure"
                proof_steps.append({
                    'step_id': step_id,
                    'natural_language': natural_language_step,
                    'lean_tactics': placeholder_tactics,
                    'lean_results': {
                        'lean_tactics': placeholder_tactics,
                        'lean_pass': False,
                        'lean_verify': False,
                        'error_msg': "Skipped due to previous verification failure"
                    },
                    'tokens': 0,
                    'time': 0.0,
                    'success_generated': False
                })
                previous_steps += f"\n  {placeholder_tactics}"
                continue
            
            # Call LLM and verify with retry for single step
            step_response, step_lean_results, step_tokens, step_time = llm_and_verify_with_retry_onestep(
                previous_steps, natural_language_step, lean_server, max_retries, selfrefine_mode
            )
            
            total_tokens += step_tokens
            total_time += step_time
            
            # Check if step processing succeeded
            if "API call failed" in step_response:
                print(f"    Step {step_id} translation failed: {step_response}")
                proof_steps.append({
                    'step_id': step_id,
                    'natural_language': natural_language_step,
                    'lean_tactics': step_response,
                    'lean_results': step_lean_results,
                    'tokens': step_tokens,
                    'time': step_time,
                    'success_generated': False
                })
                # API call failed, use placeholder tactics for continuation
                placeholder_tactics = "sorry  -- API call failed"
                previous_steps += f"\n  {placeholder_tactics}"
                # mark verification as failed, subsequent steps will be skipped
                verification_failed = True
                continue
            
            # get step code and verification result
            step_lean_tactics = step_lean_results.get('lean_tactics', '')
            is_step_pass = step_lean_results.get('lean_pass', False)
            
            if not is_step_pass:
                print(f"    Step {step_id} verification failed - marking subsequent steps as failed")
                
                # if no step code is available, use placeholder
                if not step_lean_tactics:
                    print(f"    No step code available, using placeholder")
                    step_lean_tactics = "sorry  -- verification failed"
                
                proof_steps.append({
                    'step_id': step_id,
                    'natural_language': natural_language_step,
                    'lean_tactics': step_lean_tactics,
                    'lean_results': step_lean_results,
                    'tokens': step_tokens,
                    'time': step_time,
                    'success_generated': False
                })
                
                # mark verification as failed, subsequent steps will be skipped
                verification_failed = True
            else:
                # Step succeeded, add to proof_steps
                proof_steps.append({
                    'step_id': step_id,
                    'natural_language': natural_language_step,
                    'lean_tactics': step_lean_tactics,
                    'lean_results': step_lean_results,
                    'tokens': step_tokens,
                    'time': step_time,
                    'success_generated': True
                })
                
                print(f"    Proof step {step_id} translation finished, tokens: {step_tokens}, time: {step_time:.2f}s")
            
            # no matter whether verification succeeds or fails, update previous_steps for continuation
            previous_steps += f"\n{step_lean_tactics}"
            
            # Request interval
            time.sleep(1)
        
        # Step 4: Build final Lean code and verify
        final_lean_code = previous_steps
        
        standard_header = '''import Mathlib
import Aesop

set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat Filter
'''
        # Add standard header if final_lean_code starts with "theorem"
        if final_lean_code.strip().startswith('theorem'):
            final_lean_code = standard_header + final_lean_code

        print(f"  Verifying complete Lean code...")
        try:
            lean_results = extract_code_validate(f"```lean4\n{final_lean_code}\n```", lean_server)
            is_verify_pass = lean_results.get("lean_verify", False)
            
            # add verified_code field to final verification result
            lean_results["verified_code"] = final_lean_code
            
            print(f"  Complete Lean Verification Finished: {'Success' if is_verify_pass else 'Fail'}")
            print(lean_results["error_msg"])
                
        except Exception as e:
            print(f"  Lean Verification Error: {e}")
            lean_results = {
                "lean_code": final_lean_code,
                "lean_pass": False,
                "lean_verify": False,
                "error_msg": f"Error: {str(e)}",
                "verified_code": final_lean_code
            }
        
        # Build final result
        result = {
            'id': theorem_data.get('id', f'theorem_{index + 1}'),
            'original_theorem': theorem,
            'theorem_translation': theorem_response,
            'theorem_lean_results': theorem_lean_results,
            'proof_steps': proof_steps,
            'final_lean_code': final_lean_code,
            'lean_verification': lean_results,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'num_steps': len(proof_steps),
            'generated_steps': sum(1 for step in proof_steps if step['success_generated']),
            'theorem_attempts': thm_attempt + 1
        }
        
        # Print processing result summary
        print(f"  Theorem {index + 1} processing completed")
        print(f"  Total tokens: {total_tokens}, Total time: {total_time:.2f} seconds")
        print(f"  Proof steps: {len(proof_steps)}, Successfully generated steps: {result['generated_steps']}")
        print(f"  Lean verification: {'Success' if lean_results.get('lean_verify', False) else 'Failed'}")
        print(f"  Theorem attempts: {thm_attempt + 1}")
        
        return result
    
    return None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _atomic_write_json(filepath: str, obj) -> None:
    """Safely write JSON to avoid partial/corrupt files."""
    tmp_path = f"{filepath}.tmp.{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, filepath)

def _build_save_name(item, fallback_index: int) -> str:
    """
    save_name = item['origin'].split('.')[0] + "_" + str(item['id']) + ".json"
    Uses basename for safety if 'origin' is a path.
    """
    origin = item.get("origin", "unknown.json")
    origin_base = os.path.basename(origin)
    stem = origin_base.split(".")[0]
    id_part = str(item.get("id", fallback_index))
    return f"{stem}_{id_part}.json"

def _save_item_result(item, result, output_dir: str, index: int) -> str:
    _ensure_dir(output_dir)
    filename = _build_save_name(item, index)
    filepath = os.path.join(output_dir, filename)
    _atomic_write_json(filepath, result)
    print(f"  Saved to: {filepath}")
    return filepath

def _calculate_step_verification_stats(result):
    proof_steps = result.get('proof_steps', [])
    total_steps = len(proof_steps)
    
    if total_steps == 0:
        return {
            'verified_steps_count': 0,
            'total_steps_count': 0,
            'verified_steps_percentage': 0.0
        }
    
    # Count the number of steps that pass verification (determined by success_generated field)
    verified_steps = 0
    for step in proof_steps:
        # Check if the step passes verification using the success_generated field
        if isinstance(step, dict):
            # Use success_generated field to determine if the step passes verification
            if step.get('success_generated', False):
                verified_steps += 1
    
    verified_percentage = (verified_steps / total_steps) * 100.0 if total_steps > 0 else 0.0
    
    return {
        'verified_steps_count': verified_steps,
        'total_steps_count': total_steps,
        'verified_steps_percentage': round(verified_percentage, 2)
    }


def main_step_by_step(data, output_dir, lean_server=None, max_retries=3, max_retries_thm=1, selfrefine_mode=False):
    """
    Main function - Process all theorems step by step and save results to specified folder

    Args:
        data: List containing theorem data
        output_dir: Output folder path for saving individual results
        lean_server: Lean server instance
        max_retries: Maximum retry attempts for each step (default: 3)
        max_retries_thm: Maximum retry attempts for entire theorem processing (default: 1)
        selfrefine_mode: Whether to enable self-refine mode (default: False)
    """
    print("=" * 60)
    print("Starting step-by-step batch processing of theorem data")
    print("=" * 60)
    
    if not data:
        print("No data to process")
        return

    print(f"Theorems to process: {len(data)}")
    print(f"Output directory: {output_dir}")
    print(f"Max retries per step: {max_retries}")
    print(f"Max retries per theorem: {max_retries_thm}")
    print(f"Self-refine mode: {'Enabled' if selfrefine_mode else 'Disabled'}")
    print()

    output = []
    successful_theorems = 0
    fully_verified_theorems = 0  # Number of fully verified theorems
    grand_total_tokens = 0
    grand_total_time = 0.0
    _ensure_dir(output_dir)

    # Process each theorem and save immediately
    for i, theorem_data in enumerate(data):
        filename = _build_save_name(theorem_data, i + 1)
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):  # Skip if already exists
            print(f"Skipping {filename}: file already exists")
            continue

        try:
            result = process_single_theorem_step_by_step(
                theorem_data, i, lean_server, 
                max_retries=max_retries, 
                max_retries_thm=max_retries_thm, 
                selfrefine_mode=selfrefine_mode
            )
            
            if result is not None:
                # Calculate step-level verification statistics
                step_stats = _calculate_step_verification_stats(result)
                result.update(step_stats)
                
                # Check if the theorem is fully verified
                lean_verification = result.get('lean_verification', {})
                is_fully_verified = lean_verification.get('lean_verify', False)
                result['fully_verified'] = is_fully_verified
                
                output.append(result)
                
                if result['lean_verification'].get('lean_verify', False):
                    successful_theorems += 1
                
                if is_fully_verified:
                    fully_verified_theorems += 1

                grand_total_tokens += result.get('total_tokens', 0)
                grand_total_time += result.get('total_time', 0.0)

                # Print verification status for the current theorem
                print(f"  Theorem {i + 1}: Steps {step_stats['verified_steps_count']}/{step_stats['total_steps_count']} verified ({step_stats['verified_steps_percentage']:.1f}%)")
                print(f"  Fully verified: {'Yes' if is_fully_verified else 'No'}")
                print(f"  Running total - Fully verified theorems: {fully_verified_theorems}/{i + 1}")

                _save_item_result(theorem_data, result, output_dir, i + 1)

            else:
                # Placeholder error information
                err = {
                    'id': theorem_data.get('id', f'theorem_{i + 1}'),
                    'original_theorem': theorem_data.get('nl_theorem', ''),
                    'theorem_translation': "Processing result is empty (None)",
                    'proof_steps': [],
                    'final_lean_code': "",
                    'lean_verification': {
                        "lean_pass": False,
                        "lean_verify": False,
                        "error_msg": "process_single_theorem_step_by_step returned None"
                    },
                    'total_tokens': 0,
                    'total_time': 0.0,
                    'num_steps': 0,
                    'generated_steps': 0,
                    'verified_steps_count': 0,
                    'total_steps_count': 0,
                    'verified_steps_percentage': 0.0,
                    'fully_verified': False
                }
                output.append(err)
                print(f"  Theorem {i + 1}: Processing failed")
                print(f"  Running total - Fully verified theorems: {fully_verified_theorems}/{i + 1}")
                _save_item_result(theorem_data, err, output_dir, i + 1)

        except Exception as e:
            print(f"  Exception occurred while processing theorem {i + 1}: {e}")
            error_result = {
                'id': theorem_data.get('id', f'theorem_{i + 1}'),
                'original_theorem': theorem_data.get('nl_theorem', ''),
                'theorem_translation': f"Processing exception: {str(e)}",
                'proof_steps': [],
                'final_lean_code': "",
                'lean_verification': {
                    "lean_pass": False,
                    "lean_verify": False,
                    "error_msg": f"Processing exception: {str(e)}"
                },
                'total_tokens': 0,
                'total_time': 0.0,
                'num_steps': 0,
                'generated_steps': 0,
                'verified_steps_count': 0,
                'total_steps_count': 0,
                'verified_steps_percentage': 0.0,
                'fully_verified': False
            }
            output.append(error_result)
            print(f"  Theorem {i + 1}: Exception occurred")
            print(f"  Running total - Fully verified theorems: {fully_verified_theorems}/{i + 1}")
            _save_item_result(theorem_data, error_result, output_dir, i + 1)

        if i < len(data) - 1:
            print("  Waiting 1 second...")
            time.sleep(1)
    
    # Calculate overall step verification statistics
    total_all_steps = sum(result.get('total_steps_count', 0) for result in output)
    total_verified_steps = sum(result.get('verified_steps_count', 0) for result in output)
    overall_step_percentage = (total_verified_steps / total_all_steps) * 100.0 if total_all_steps > 0 else 0.0
    
    # Print final statistics
    print("=" * 60)
    print("Batch processing completed")
    print("=" * 60)
    print(f"Total theorems processed: {len(output)}")
    print(f"Successfully verified theorems: {successful_theorems}")
    print(f"Fully verified theorems: {fully_verified_theorems}")  
    print(f"Success rate: {successful_theorems/len(output)*100:.1f}%" if output else "N/A")
    print(f"Full verification rate: {fully_verified_theorems/len(output)*100:.1f}%" if output else "N/A")  
    print(f"Total steps across all theorems: {total_all_steps}")  
    print(f"Total verified steps: {total_verified_steps}")  
    print(f"Overall step verification rate: {overall_step_percentage:.1f}%")  
    print(f"Total tokens used: {grand_total_tokens}")
    print(f"Total time elapsed: {grand_total_time:.2f} seconds")
    print(f"Average tokens per theorem: {grand_total_tokens/len(output):.1f}" if output else "N/A")
    print(f"Average time per theorem: {grand_total_time/len(output):.2f} seconds" if output else "N/A")
    
    return output


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Script with dry-run option')

    # Add arguments
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode (no actual changes will be made)')

    # Parse arguments
    args = parser.parse_args()

    dry_run = args.dry_run

    # Load data    
    with open("data/benchmark_0409.json", 'r') as file:
        data = json.load(file)

    # Display the dry-run status
    if dry_run:
        print("Running in DRY-RUN mode - test one data sample only")
        data = data[:1]
        output_dir = "benchmark_results/output_test/benchmark_stepproof_think_test"
    else:
        output_dir = "benchmark_results/output_pickle/benchmark_stepproof_think"
    
    main_step_by_step(data, output_dir = output_dir,  lean_server=lean_server,max_retries=5, max_retries_thm=1, selfrefine_mode=True)   
    print("Done")