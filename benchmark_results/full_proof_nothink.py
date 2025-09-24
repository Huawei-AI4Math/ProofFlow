from dotenv import load_dotenv
from proofflow import LeanServer
from typing import List, Dict, Any, Tuple
from openai import OpenAI
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

# Thinking mode is disabled
MODEL_NAME="google/gemini-2.5-flash"
THINKING_MODE = False


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
                extra_headers={
                    "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
                    "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
                },
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

FORMALIZATION_PROMPT = """
You are a Lean 4 theorem proving specialist with expertise in formal mathematics and proof assistant systems. Your task is to translate informal mathematical proofs into rigorous, syntactically correct Lean 4 formal proofs.

## Instructions:
1. First translate the informal problem into a formal theorem statement, identify all mathematical objects, hypotheses and the proof goal.
2. Carefully analyze the informal proof structure, logical flow and proof steps.
3. Apply appropriate Lean 4 tactics to ensure syntax correctness.
4. Maintain mathematical rigor while preserving the original proof's essence.
5. Keep the header of the Lean 4 code unchanged.


## Current Formalization Task:
### Informal Problem Statement:
{theorem}

### Informal Proof:
{proof}

### Target Formal Problem Statement and Proof:
```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

(output the Lean 4 code here)
```

Output the Lean 4 code only, wrap the code between "```lean4" and "```".
"""

REFINE_PROMPT =  """
You are a Lean 4 theorem proving specialist with expertise in formal mathematics and proof assistant systems. Your task is to translate informal mathematical proofs into rigorous, syntactically correct Lean 4 formal proofs.

## Instructions:
1. First translate the informal problem into a formal theorem statement, identify all mathematical objects, hypotheses and the proof goal.
2. Carefully analyze the informal proof structure, logical flow and proof steps.
3. Apply appropriate Lean 4 tactics to ensure syntax correctness.
4. Maintain mathematical rigor while preserving the original proof's essence.
5. Keep the header of the Lean 4 code unchanged.


## Current Formalization Task:
### Informal Problem Statement:
{theorem}

### Informal Proof:
{proof}

### Target Formal Problem Statement and Proof:
```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

{last_lean_code}
```

The previous Lean4 code I sent you contains errors. Please take that into account.
Lean error/warnings: {error_massage} 

Based on these errors, please correct the previous response. Output the Lean 4 code only, wrap the code between "```lean4" and "```".
"""

def full_proof_formalization(theorem: str, proof: str, self_refine: bool = False, last_lean_code: str = "", error_massage: str = "") -> Tuple[str, int, float]:
    """
    Formalize theorem and proof
    Returns: (response content, output token count, call time)
    """
    if self_refine:
        prompt = REFINE_PROMPT.format(
            theorem=theorem.strip(),
            proof=proof.strip(),
            last_lean_code=last_lean_code.strip(),
            error_massage=error_massage.strip()
        )
    else:
        prompt = FORMALIZATION_PROMPT.format(
            theorem=theorem.strip(),
            proof=proof.strip()
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
    # response = process_lean_string(response) #add missing imports

    try:
        lean_pass, lean_verify, error_msg = lean_server.check_lean_string(response)
        if not isinstance(error_msg, str):
            error_msg = str(error_msg)
    except Exception as e:
        raise RuntimeError("Error during Lean code verification " + str(e))
    
    return {"lean_code": response, "lean_pass": lean_pass, "lean_verify": lean_verify, "error_msg": error_msg} #None if lean_verify else error_msg


def llm_and_verify_with_retry(theorem: str, proof: str, lean_server, max_verify_retries: int, selfrefine_mode: bool = False) -> Tuple[str, Dict[str, Any], int, float]:
    """
    Complete retry workflow for LLM generation + Lean verification
    When verification fails, it will recall LLM to generate new response and verify again
    
    Args:
        theorem: Theorem content
        proof: Proof content
        lean_server: Lean server instance
        max_verify_retries: Maximum verification retry attempts
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
    
    for verify_attempt in range(max_verify_retries):
        print(f"  Round {verify_attempt + 1}/{max_verify_retries}: LLM generation + Lean verification...")
        
        # Step 1: Call LLM to generate response
        # For first attempt or non-self-refine mode, use original calling method
        if verify_attempt == 0 or not selfrefine_mode:
            llm_response, output_tokens, call_time = full_proof_formalization(theorem, proof)
        else:
            # Self-refine mode: pass previous code and error information

            print("testtest:")
            print(last_error_msg)

            llm_response, output_tokens, call_time = full_proof_formalization(
                theorem, proof, 
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
                "tries": verify_attempt + 1,
                "attempt_history": attempt_history,
                "total_tokens": total_tokens,
                "total_time": total_time,
                "avg_tokens_per_trial": total_tokens / (verify_attempt + 1) if verify_attempt >= 0 else 0,
                "avg_time_per_trial": total_time / (verify_attempt + 1) if verify_attempt >= 0 else 0
            }
            return final_llm_response, lean_results, total_tokens, total_time
        
        print(f"  LLM generation successful, tokens: {output_tokens}, time: {call_time:.2f}s")
        if selfrefine_mode and verify_attempt > 0:
            print(f"  Using self-refine mode, improving based on previous error")
        
        # Step 2: Perform Lean verification
        try:
            print(f"  Starting Lean verification...")
            lean_results = extract_code_validate(llm_response, lean_server)

            # Record this attempt
            attempt_record = {
                "attempt": verify_attempt + 1,
                "llm_response": llm_response,
                "lean_code": lean_results.get("lean_code", ""),
                "lean_pass": lean_results.get("lean_pass", False),
                "lean_verify": lean_results.get("lean_verify", False),
                "error_msg": lean_results.get("error_msg", ""),
                "tokens": output_tokens,
                "time": call_time,
                "used_selfrefine": selfrefine_mode and verify_attempt > 0
            }
            attempt_history.append(attempt_record)
            
            # If verification succeeds, return results
            if lean_results.get("lean_verify", False):
                print(f"  Lean verification successful!")
                lean_results["tries"] = verify_attempt + 1
                lean_results["attempt_history"] = attempt_history
                lean_results["total_tokens"] = total_tokens
                lean_results["total_time"] = total_time
                lean_results["avg_tokens_per_trial"] = total_tokens / (verify_attempt + 1)
                lean_results["avg_time_per_trial"] = total_time / (verify_attempt + 1)
                return final_llm_response, lean_results, total_tokens, total_time
            else:
                # Save error info and code for next self-refine use
                last_error_msg = lean_results.get("error_msg", "Unknown error")
                last_lean_code = lean_results.get("lean_code", "")
                print(f"  Lean verification failed: {last_error_msg}")
                
                if selfrefine_mode and verify_attempt < max_verify_retries - 1:
                    print(f"  Will use self-refine mode for improvement in next attempt")
                
        except Exception as e:
            error_msg = f"Verification exception: {str(e)}"
            print(f"  Lean verification exception: {e}")
            
            # Save exception info for self-refine use
            last_error_msg = error_msg
            last_lean_code = ""  # May not have valid code in exception case
            
            attempt_history.append({
                "attempt": verify_attempt + 1,
                "llm_response": llm_response,
                "lean_code": "",
                "lean_pass": False,
                "lean_verify": False,
                "error_msg": error_msg,
                "tokens": output_tokens,
                "time": call_time,
                "used_selfrefine": selfrefine_mode and verify_attempt > 0
            })
        
        # If not the last attempt, wait and continue to next round
        if verify_attempt < max_verify_retries - 1:
            if selfrefine_mode:
                print(f"  Verification failed, will regenerate LLM response using self-refine mode...")
            else:
                print(f"  Verification failed, will regenerate LLM response...")
            time.sleep(1)
    
    # All verification attempts failed
    print(f"  Final failure: attempted {max_verify_retries} rounds of LLM generation + verification")
    lean_results = {
        "lean_code": last_lean_code,
        "lean_pass": False, 
        "lean_verify": False, 
        "error_msg": f"Verification failed after {max_verify_retries} retries",
        "tries": max_verify_retries,
        "attempt_history": attempt_history,
        # Added: total and average data
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tokens_per_trial": total_tokens / max_verify_retries if max_verify_retries > 0 else 0,
        "avg_time_per_trial": total_time / max_verify_retries if max_verify_retries > 0 else 0
    }
    return final_llm_response, lean_results, total_tokens, total_time

def process_single_item(item: Dict[str, Any], index: int, max_verify_retries: int, lean_server, selfrefine_mode: bool = False) -> Dict[str, Any]:
    """
    Process single data item
    
    Args:
        item: Single data item
        index: Data index
        max_verify_retries: Maximum retry attempts (each retry will call LLM + verification again)
        lean_server: Lean server instance
        selfrefine_mode: Whether to enable self-refine mode
    
    Returns:
        dict: Processing result, returns None if failed
    """
    print(f"Processing item {index + 1}...")
    
    # Extract theorem and proof
    theorem = item.get('nl_theorem', '')
    proof = item.get('nl_proof', '')
    
    if not theorem or not proof:
        print(f"  Skipping item {index + 1} (missing required fields)")
        return None
    
    # Execute complete retry workflow of LLM generation + Lean verification
    final_llm_response, lean_results, total_tokens, total_time = llm_and_verify_with_retry(
        theorem, proof, lean_server, max_verify_retries, selfrefine_mode
    )
    
    # Construct output result
    result = {
        'id': item.get('id', f'item_{index + 1}'),
        'original_theorem': theorem,
        'original_proof': proof,
        'LLM_output': final_llm_response,
        'Lean_results': lean_results,
        'total_tokens': total_tokens,  # Total tokens for all retries of this problem
        'total_time': total_time,      # Total time for all retries of this problem
        'avg_tokens_per_trial': lean_results.get('avg_tokens_per_trial', 0),  # Average tokens per trial
        'avg_time_per_trial': lean_results.get('avg_time_per_trial', 0)       # Average time per trial
    }
    
    # Print processing result summary
    print(f"  Item {index + 1} processing completed")
    print(f"  Total tokens: {total_tokens}, Total time: {total_time:.2f}s")
    print(f"  Avg tokens per trial: {result['avg_tokens_per_trial']:.1f}, Avg time per trial: {result['avg_time_per_trial']:.2f}s")
    print(f"  Retry attempts: {lean_results.get('tries', 0)}")
    
    lean_code = lean_results.get("lean_code", "")
    if lean_code:
        print(f"  Successfully extracted Lean code, length: {len(lean_code)} characters")
        if lean_results.get("lean_verify", False):
            print(f"  Lean verification successful ✓")
        else:
            print(f"  Lean verification failed ✗")
    else:
        print("  Warning: Failed to extract Lean code from LLM output")
    
    return result


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _atomic_write_json(filepath: str, obj) -> None:
    """
    Write JSON atomically to avoid corrupt/partial files on interruptions.
    """
    tmp_path = f"{filepath}.tmp.{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, filepath)

def _build_save_name(item, fallback_index: int) -> str:
    """
    save_name = item['origin'].split('.')[0] + "_" + str(item['id']) + ".json"
    Use basename for safety if 'origin' is a path; otherwise mirrors your rule.
    """
    origin = item.get("origin", "unknown.json")
    # Keep behavior close to your split rule but safer if origin contains path parts
    origin_base = os.path.basename(origin)
    stem = origin_base.split(".")[0]  # matches your requirement
    id_part = str(item.get("id", fallback_index))
    return f"{stem}_{id_part}.json"

def _save_item_result(item, result, output_dir: str, index: int) -> str:
    _ensure_dir(output_dir)
    filename = _build_save_name(item, index + 1)
    filepath = os.path.join(output_dir, filename)
    _atomic_write_json(filepath, result)
    print(f"  Saved to: {filepath}")
    return filepath

def main(data, output_dir, max_verify_retries: int = 2, lean_server=None, selfrefine_mode: bool = False):
    """
    Main function - process all data items and save results to specified folder one by one

    Args:
        output_dir: Output folder path for saving individual results
        max_verify_retries: Maximum retry attempts for Lean verification (default 2)
        lean_server: Lean server instance
        selfrefine_mode: Whether to enable self-refine mode (default False)
    """
    print("=" * 60)
    print("Starting batch data processing")
    print("=" * 60)
    
    if not data:
        print("No data to process")
        return

    print(f"Data to process: {len(data)} items")
    print(f"Maximum Lean verification retries: {max_verify_retries}")
    print(f"Self-refine mode: {'Enabled' if selfrefine_mode else 'Disabled'}")
    print(f"Output directory: {output_dir}")
    print()

    output = []  # Keep in-memory summary for statistics only
    grand_total_tokens = 0
    grand_total_time = 0.0
    successful_items = 0

    _ensure_dir(output_dir)

    # Process each data item and save immediately
    for i, item in enumerate(data):
        filename = _build_save_name(item, i + 1)
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):  # Check if file exists
            print(f"Skipping {filename}: file already exists")
            continue
        try:
            result = process_single_item(item, i, max_verify_retries, lean_server, selfrefine_mode)
            
            if result is not None:
                output.append(result)
                successful_items += 1

                # Update total statistics
                grand_total_tokens += result.get('total_tokens', 0)
                grand_total_time += result.get('total_time', 0.0)

                # —— Key change: save item by item —— #
                _save_item_result(item, result, output_dir, i)
            
            else:
                # Even if returns None, save placeholder error info to avoid data loss
                err = {
                    'id': item.get('id', f'item_{i + 1}'),
                    'original_theorem': item.get('nl_theorem', ''),
                    'original_proof': item.get('nl_proof', ''),
                    'LLM_output': "Processing result is empty (None)",
                    'Lean_results': {
                        "lean_code": "",
                        "lean_pass": False, 
                        "lean_verify": False, 
                        "error_msg": "process_single_item returned None",
                        "tries": 0,
                        "attempt_history": [],
                        "total_tokens": 0,
                        "total_time": 0.0,
                        "avg_tokens_per_trial": 0,
                        "avg_time_per_trial": 0
                    },
                    'total_tokens': 0,
                    'total_time': 0.0,
                    'avg_tokens_per_trial': 0,
                    'avg_time_per_trial': 0
                }
                output.append(err)
                _save_item_result(item, err, output_dir, i)

        except Exception as e:
            print(f"  Exception occurred while processing item {i + 1}: {e}")
            # Record and save even when exception occurs to avoid data loss
            error_result = {
                'id': item.get('id', f'item_{i + 1}'),
                'original_theorem': item.get('nl_theorem', ''),
                'original_proof': item.get('nl_proof', ''),
                'LLM_output': f"Processing exception: {str(e)}",
                'Lean_results': {
                    "lean_code": "",
                    "lean_pass": False, 
                    "lean_verify": False, 
                    "error_msg": f"Processing exception: {str(e)}",
                    "tries": 0,
                    "attempt_history": [],
                    "total_tokens": 0,
                    "total_time": 0.0,
                    "avg_tokens_per_trial": 0,
                    "avg_time_per_trial": 0
                },
                'total_tokens': 0,
                'total_time': 0.0,
                'avg_tokens_per_trial': 0,
                'avg_time_per_trial': 0
            }
            output.append(error_result)
            _save_item_result(item, error_result, output_dir, i)
        
            # Print final statistics
    print("=" * 60)
    print("Batch processing completed")
    print("=" * 60)
    
    total_processed = len(output)
    successful_verifications = sum(1 for result in output if result.get('Lean_results', {}).get('lean_verify', False))
    failed_verifications = total_processed - successful_verifications
    
    print(f"Total items processed: {total_processed}")
    print(f"Successfully verified: {successful_verifications}")
    print(f"Failed verification: {failed_verifications}")
    if total_processed > 0:
        success_rate = (successful_verifications / total_processed) * 100
        print(f"Success rate: {success_rate:.2f}%")
    
    print(f"Total tokens consumed: {grand_total_tokens}")
    print(f"Total time elapsed: {grand_total_time:.2f}s")
    if total_processed > 0:
        print(f"Average tokens per item: {grand_total_tokens / total_processed:.1f}")
        print(f"Average time per item: {grand_total_time / total_processed:.2f}s")
    
    print("=" * 60)


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
        output_dir = "benchmark_results/output_test/benchmark_fullproof_nothink_test"
    else:
        output_dir = "benchmark_results/output_pickle/benchmark_fullproof_nothink"

    main(data, output_dir=output_dir, max_verify_retries=5, lean_server=lean_server, selfrefine_mode=True)
    print("Done")