import copy
import os
import subprocess
import sys
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx
import tiktoken
import torch
from accelerate import Accelerator
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- Utilities ----------
def remove_imports(lean_code):
    # Lines to remove from individual code blocks
    lines_to_remove = [
        "import Mathlib",
        "import Aesop",
        "set_option maxHeartbeats 0",
        "open BigOperators Real Nat Topology Rat Filter",
    ]
    code_lines = lean_code.split("\n")
    filtered_lines = []
    for line in code_lines:
        line_stripped = line.strip()
        if line_stripped not in lines_to_remove:
            filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def count_tokens(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += len(encoding.encode(msg["content"]))
    return total


def trim_to_limit(messages, max_tokens=1000000000):
    while count_tokens(messages) > max_tokens and len(messages) > 4:
        messages.pop(
            1
        )  # Remove oldest non-system message until there is only 5 messages left
    return messages


def get_ready_items(dependencies: Dict[Any, set], completed: set, futures: set):
    """
    Returns a list of item_ids that are ready to be processed (all dependencies satisfied).
    """
    return [
        item_id
        for item_id, deps in dependencies.items()
        if deps <= completed and item_id not in completed and item_id not in futures
    ]


def extract_answer_openai(response) -> Tuple[bool, str]:
    """
    Robustly extract the first message content from an OpenAI ChatCompletion response.
    """
    try:
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            content = getattr(msg, "content", None)
            if content:
                return True, content
            else:
                return False, "Empty content in response."
        return False, "No choices in response."
    except Exception as e:
        return False, f"Failed to parse response: {e}"


# ---------- Main Class for LLM Management ----------


class LLMManager:
    """
    A comprehensive Large Language Model manager supporting both remote APIs and local models.
    
    The LLMManager class provides a unified interface for interacting with various
    Large Language Models, supporting both remote API calls (OpenAI-compatible) and
    local model inference using Hugging Face Transformers. It handles tokenization,
    generation, logging, and error management automatically.
    
    Features:
        - Remote API support (OpenAI, Anthropic, custom endpoints)
        - Local model support via Hugging Face Transformers
        - Automatic token counting and message trimming
        - Comprehensive logging and error handling
        - System prompt management
        - Configurable generation parameters
        
    Supported Models:
        - Remote: Any OpenAI-compatible API (GPT-4, Claude, etc.)
        - Local: Any Hugging Face causal language model
        
    Example:
        >>> # Remote API usage
        >>> model_info = {
        ...     "model": "gpt-4",
        ...     "base_url": "https://api.openai.com/v1",
        ...     "api_key": "your-api-key"
        ... }
        >>> llm = LLMManager(model_info, system_prompt_path="prompt.txt")
        >>> response, messages = llm.call_llm([{"role": "user", "content": "Hello!"}])
        
        >>> # Local model usage
        >>> model_info = {"model": "microsoft/DialoGPT-medium"}
        >>> llm = LLMManager(model_info)
        >>> response, messages = llm.call_llm([{"role": "user", "content": "Hello!"}])
    """
    
    def __init__(
        self, model_info: Dict[str, Any], system_prompt_path: str = None, **kwargs
    ):
        """
        Initialize the LLMManager with model configuration and optional system prompt.

        Args:
            model_info (Dict[str, Any]): Model configuration dictionary. Must contain:
                - For remote APIs: "model" (str), "base_url" (str), "api_key" (str, optional)
                - For local models: "model" (str) - Hugging Face model name/path
            system_prompt_path (str, optional): Path to a file containing the system prompt.
                If provided, this prompt will be automatically prepended to all conversations.
            **kwargs: Additional arguments passed to the model client or Hugging Face model.

        Raises:
            ImportError: If required dependencies are missing for local models.
            RuntimeError: If model loading fails.
            FileNotFoundError: If system_prompt_path is provided but file doesn't exist.

        Example:
            >>> # Remote API configuration
            >>> model_info = {
            ...     "model": "gpt-4",
            ...     "base_url": "https://api.openai.com/v1",
            ...     "api_key": "sk-..."
            ... }
            >>> llm = LLMManager(model_info, system_prompt_path="system.txt")
            
            >>> # Local model configuration
            >>> model_info = {"model": "microsoft/DialoGPT-medium"}
            >>> llm = LLMManager(model_info, torch_dtype=torch.float16)
        """
        self.model_info = model_info
        self.tokenizer = None
        self.model = None
        self.client = None
        self.system_prompt = (
            open(system_prompt_path, "r", encoding="utf-8").read()
            if system_prompt_path
            else None
        )

        if "base_url" in self.model_info:
            # Remote API Path: Initialize the OpenAI client once
            base_url = self.model_info["base_url"]
            self.client = OpenAI(
                base_url=base_url,
                api_key=self.model_info.get("api_key"),
                http_client=httpx.Client(
                    base_url=base_url,
                    follow_redirects=True,
                    verify=False,
                    timeout=60 * 20,
                ),
                **kwargs,
            )
        else:
            # Local Transformers Path: Load the model and tokenizer once
            try:
                model_name_or_path = self.model_info["model"]
                print(f"Loading local model with Hugging Face: {model_name_or_path}...")

                # Initialize Accelerate
                self.accelerator = Accelerator()

                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    **kwargs,
                )

                # Prepare model for inference with Accelerate
                self.model = self.accelerator.prepare(self.model)
                self.model.eval()
                print("Model loaded successfully.")
            except ImportError as e:
                raise ImportError(
                    f"Hugging Face Transformers, Accelerate, and/or PyTorch are required for local models: {e}"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load local model: {e}")

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        logs: Optional[list] = None,
        max_new_tokens: int = 16384,
        temperature: float = 0.9,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Generate a response from the configured LLM using the provided messages.

        This method handles both remote API calls and local model inference,
        automatically managing tokenization, generation parameters, and logging.
        The system prompt (if provided) is automatically prepended to the conversation.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.
                Supported roles: 'system', 'user', 'assistant'.
                Example: [{"role": "user", "content": "Hello!"}]
            logs (list, optional): List to append detailed logging information.
                Each log entry contains timing, model info, tokens, and success status.
            max_new_tokens (int, optional): Maximum number of new tokens to generate.
                Defaults to 16384. Only applies to local models.
            temperature (float, optional): Sampling temperature for generation.
                Higher values (e.g., 0.9) make output more random, lower values (e.g., 0.1) more focused.
                Defaults to 0.9.

        Returns:
            Tuple[str, List[Dict[str, str]]]: A tuple containing:
                - response_text (str): The generated response text
                - updated_messages (List[Dict[str, str]]): The original messages with the
                  assistant's response appended

        Example:
            >>> llm = LLMManager(model_info)
            >>> messages = [{"role": "user", "content": "What is 2+2?"}]
            >>> response, updated_messages = llm.call_llm(messages)
            >>> print(response)  # "2+2 equals 4"
            
        Note:
            - For remote APIs, token limits are automatically applied based on model
            - For local models, max_new_tokens controls generation length
            - System prompts are automatically prepended if provided during initialization
            - All API calls are logged with timing and token information
        """
        start_time = time.time()
        success_flag = False
        result_text = ""
        response_payload = None
        model_name = self.model_info["model"]
        messages = copy.deepcopy(messages)
        new_tokens = None

        # Check if the first message is already the system prompt
        if self.system_prompt:
            if not messages or messages[0].get("role") != "system":
                # Insert the system prompt at the beginning of the list
                messages.insert(0, {"role": "system", "content": self.system_prompt})

        try:
            if self.client:
                # ----- Remote (OpenAI-compatible) path -----
                token_limit = 40000 if "goedel" in model_name.lower() else 300000
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=trim_to_limit(messages, token_limit),
                    temperature=temperature,
                )
                success_flag, result_text = extract_answer_openai(completion)
                response_payload = result_text if success_flag else completion
                new_tokens = completion.usage.completion_tokens

            elif self.model and self.tokenizer:
                # ----- Local (Hugging Face) path -----

                # Tokenize the chat messages
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                )

                # Move inputs to the correct device
                inputs = self.accelerator.prepare(inputs)

                # Generate a response from the model
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                    )

                # Decode the generated tokens
                result_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )

                # Find the start of the assistant's response to exclude the prompt
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                if prompt_text in result_text:
                    result_text = result_text.split(prompt_text, 1)[-1].strip()

                success_flag = True
                response_payload = result_text

            else:
                raise RuntimeError("LLMManager was not initialized correctly.")

        except Exception as e:
            success_flag = False
            result_text = f"Request failed: {e}"
            response_payload = str(e)
            print(f"Error during API call: {e}")

        end_time = time.time()

        # add assistant response to messages
        messages.append({"role": "assistant", "content": response_payload})

        if logs is not None:
            logs.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "model": model_name,
                    "messages": messages,
                    "response": response_payload,
                    "success": success_flag,
                    "base_url": self.model_info.get("base_url"),
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "generated_tokens": new_tokens,
                }
            )

        # print(result_text)

        # UNCOMMENT LATER!
        # if success_flag:
        return result_text, messages
        # else:
        #    raise ConnectionError("Could not get answer from LLM.")


# ==============================================================================
# VLLM Python Runner
# This script launches two VLLM API servers using Python's subprocess module.
# Each function targets a specific model, GPU, and port, making it easy to manage.
# ==============================================================================


def start_vllm_server(model_path: str, cuda_device: int, port: int, log_file: str):
    """
    Launch a VLLM API server in the background for high-performance model serving.

    This function starts a VLLM (Very Large Language Model) server that provides
    an OpenAI-compatible API endpoint for serving large language models efficiently.
    The server runs in the background and can be used by LLMManager for inference.

    Args:
        model_path (str): The full path to the VLLM model directory or model name.
            Can be a local path or Hugging Face model identifier.
        cuda_device (int): The CUDA device number to run the model on.
            Example: 0, 1, 2, 3 for different GPUs.
        port (int): The port number for the API server to listen on.
            Example: 8000, 8001, 14457.
        log_file (str): The file path to redirect standard output and error logs.
            Example: "/path/to/vllm_server.log".

    Raises:
        FileNotFoundError: If Python executable or vLLM module is not found.
        Exception: If the server fails to start for any other reason.

    Example:
        >>> start_vllm_server(
        ...     model_path="/path/to/llama-2-7b-chat",
        ...     cuda_device=0,
        ...     port=8000,
        ...     log_file="vllm_server.log"
        ... )
        >>> # Server is now running and can be accessed at http://localhost:8000
        
    Note:
        - The server runs in the background using subprocess.Popen
        - CUDA_VISIBLE_DEVICES is automatically set for the subprocess
        - Server configuration includes tensor parallelism and memory optimization
        - Use the returned process ID to monitor or terminate the server
    """
    print(
        f"Launching model '{os.path.basename(model_path)}' on GPU {cuda_device} on port {port}..."
    )

    # Set the CUDA_VISIBLE_DEVICES environment variable for this subprocess call.
    # The shell script exports this, but for a subprocess, we can set it
    # for just this specific command.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # Construct the command as a list of arguments.
    # This is generally safer than using a single shell string.
    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--tensor-parallel-size",
        "1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.95",  # Slightly increased memory utilization
    ]

    try:
        # Use subprocess.Popen to run the command in the background.
        # This is the Python equivalent of 'nohup ... &'.
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,  # Redirect stderr to the same log file
                text=True,  # Use text mode for file I/O
            )
        print(f"VLLM server for '{os.path.basename(model_path)}' started.")
        print(f"Logs are being written to {log_file}")
        print(f"Process ID (PID): {process.pid}")

    except FileNotFoundError:
        print(
            f"Error: Python executable or vLLM module not found. "
            f"Please ensure your environment is configured correctly.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
