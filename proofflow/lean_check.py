import ast
import os
import re
import subprocess
from pathlib import Path

from kimina_client import (
    CheckResponse,
    KiminaClient,
    ReplResponse,
    Snippet,
    SnippetStatus,
)


def extract_errors(text):
    errors = []
    # find all `response={...}` substrings
    for match in re.finditer(r"response=(\{.*?\}) diagnostics=None", text, re.DOTALL):
        response_str = match.group(1)
        try:
            # make it valid Python dict (single quotes -> double)
            response_str_jsonlike = response_str.replace("null", "None")
            response = ast.literal_eval(response_str_jsonlike)
            for msg in response.get("messages", []):
                if msg.get("severity") == "error":
                    errors.append(
                        {
                            "line": msg["pos"]["line"],
                            "column": msg["pos"]["column"],
                            "endLine": msg["endPos"]["line"],
                            "endColumn": msg["endPos"]["column"],
                            "data": msg["data"],
                        }
                    )
        except Exception as e:
            print("parse failed:", e)
    return errors


# --- Provided helper functions from the user's prompt ---

LEAN_LIBRARIES = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter"""


def process_lean_string(lean_string: str):
    """
    Process a Lean code string to ensure required imports and options are present.

    Adds the following in order if missing:
    1. import Mathlib (first line)
    2. import Aesop (second line, after Mathlib)
    3. set_option maxHeartbeats 0 (after Aesop)
    4. open BigOperators Real Nat Topology Rat Filter (after set_option)

    Args:
        lean_string (str): The Lean code as a string

    Returns:
        str: The processed Lean code with required imports/options added
    """

    # Split into lines and remove empty lines at the start for processing
    lines = lean_string.split("\n")

    # Required statements to check for
    required_statements = [
        "import Mathlib",
        "import Aesop",
        "set_option maxHeartbeats 0",
        "open BigOperators Real Nat Topology Rat Filter",
    ]

    # Check which statements are already present
    present_statements = set()
    for line in lines:
        line_stripped = line.strip()
        for stmt in required_statements:
            if stmt in line_stripped:
                present_statements.add(stmt)

    # Find the insertion point (after any existing imports/options)
    insert_index = 0

    # Skip any existing imports or set_option statements at the beginning
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if (
            line_stripped.startswith("import ")
            or line_stripped.startswith("set_option ")
            or line_stripped.startswith("open ")
            or line_stripped == ""
            or line_stripped.startswith("--")
        ):  # Also skip comments and empty lines
            insert_index = i + 1
        else:
            break

    # Build list of statements to insert
    statements_to_insert = []

    for stmt in required_statements:
        if stmt not in present_statements:
            statements_to_insert.append(stmt)

    # Insert missing statements at the appropriate position
    if statements_to_insert:
        # Insert in reverse order to maintain correct positioning
        for stmt in reversed(statements_to_insert):
            lines.insert(insert_index, stmt)

    # Ensure there's an empty line after the imports/options section
    # Find the end of the imports/options section
    imports_end_index = 0
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if (
            line_stripped.startswith("import ")
            or line_stripped.startswith("set_option ")
            or line_stripped.startswith("open ")
            or line_stripped.startswith("--")
        ):  # Skip comments too
            imports_end_index = i + 1
        elif line_stripped == "":  # Empty line
            continue
        else:
            break

    # Check if there's already an empty line after imports
    if (
        imports_end_index < len(lines)
        and imports_end_index > 0
        and lines[imports_end_index].strip() != ""
    ):
        lines.insert(imports_end_index, "")

    # Join lines back together
    result = "\n".join(lines)

    # Clean up any duplicate empty lines at the start
    while result.startswith("\n"):
        result = result[1:]

    return result


# create a function that picks a lean_strings and does the following
# check if imports are missing by the following steps:
# if "import Mathlib" does not exist add it as first line
# if "import Aesop" does not exists add it second line after Mathlib
# same thing for this set_option maxHeartbeats 0 -> add it after aesor
# same thing for this open BigOperators Real Nat Topology Rat


def verify_lean_lemma_server(
    lean_string: str, client: KiminaClient, add_imports=False, timeout: int = 180
):
    """
    Verifies a Lean lemma using a remote server API.
    The client object is now passed to this function.
    """
    full_code = f"{LEAN_LIBRARIES}\n\n{lean_string}" if add_imports else lean_string

    snippets = [
        Snippet(id=str(idx), code=proof) for idx, proof in enumerate([full_code])
    ]

    timeout = 60
    compilation_result: CheckResponse = client.check(
        snips=snippets,
        timeout=timeout,
        max_workers=10,
    )

    results: list[ReplResponse] = compilation_result.results

    output = str(results[0])

    # Check for errors
    error_patterns = [
        r'"severity"\s*:\s*"error"',  # Double quotes with optional spaces
        r"'severity'\s*:\s*'error'",  # Single quotes with optional spaces
    ]

    lean_pass = not any(re.search(pattern, output) for pattern in error_patterns)

    # Check for verification (no errors, no sorries, no failures)
    lean_verify = lean_pass and not any(
        [
            "declaration uses 'sorry'" in output,
            'declaration uses "sorry"' in output,
            "failed" in output,
        ]
    )

    output = extract_errors(output)

    return lean_pass, lean_verify, output


def verify_lean_lemma_local(
    lean_string: str, project_path: str, add_imports=False, timeout: int = 180
):
    """
    Verifies a single Lean lemma or theorem using the `lean` executable.
    (Function body as provided in the user's prompt)
    """
    # Create a temporary directory and file
    temp_dir = Path(project_path) / "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = temp_dir / f"temp.lean"
    full_code = f"{LEAN_LIBRARIES}\n\n{lean_string}" if add_imports else lean_string

    try:
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        result = subprocess.run(
            ["lake", "env", "lean", temp_file_path, "--json"],
            cwd=project_path,
            capture_output=True,
            text=True,
            encoding="utf-8",  # Force UTF-8 encoding
            errors="replace",  # Replace any problematic characters
            timeout=timeout,
        )

        # Check if process failed
        if result.returncode != 0 and not result.stdout:
            return (
                False,
                False,
                f"Process failed with exit code {result.returncode}: {result.stderr.strip()}",
            )

        # Check if no output (clean file)
        if not result.stdout.strip():
            return True, True, None

        # Parse the output
        output = result.stdout

        # Check for errors
        error_patterns = [
            r'"severity"\s*:\s*"error"',  # Double quotes with optional spaces
            r"'severity'\s*:\s*'error'",  # Single quotes with optional spaces
        ]

        lean_pass = not any(re.search(pattern, output) for pattern in error_patterns)

        # Check for verification (no errors, no sorries, no failures)
        lean_verify = lean_pass and not any(
            [
                "declaration uses 'sorry'" in output,
                'declaration uses "sorry"' in output,
                "failed" in output,
            ]
        )

        return lean_pass, lean_verify, output

    except FileNotFoundError:
        return (
            False,
            False,
            "Lean executable not found. Make sure it's in your system's PATH.",
        )
    except Exception as e:
        return (False, False, f"Verification failed with exception: {e}")

    finally:
        # 5. Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- The LeanServer class ---


class LeanServer:
    """
    A comprehensive Lean 4 verification server that supports both remote API and local execution.
    
    The LeanServer class provides a unified interface for verifying Lean 4 code snippets
    using either a remote Kimina server API or a local Lean installation. It automatically
    handles imports, error detection, and verification status reporting.
    
    Features:
        - Remote server verification via Kimina API
        - Local Lean 4 verification using lake/lean executables
        - Automatic import management (Mathlib, Aesop, etc.)
        - Comprehensive error detection and reporting
        - Support for both compilation and verification checks
        
    Modes:
        - Server Mode: Uses remote Kimina API for verification
        - Local Mode: Uses local Lean 4 installation for verification
        
    Example:
        >>> # Server mode
        >>> lean_server = LeanServer(api_url="http://localhost:14457")
        >>> lean_pass, lean_verify, errors = lean_server.check_lean_string("theorem test : 1 + 1 = 2 := by rfl")
        
        >>> # Local mode  
        >>> lean_server = LeanServer(project_path="/path/to/lean/project")
        >>> lean_pass, lean_verify, errors = lean_server.check_lean_string("theorem test : 1 + 1 = 2 := by rfl")
    """

    def __init__(self, api_url: str = None, project_path: str = None):
        """
        Initialize the LeanServer in either server or local mode.

        Args:
            api_url (str, optional): The URL for the remote Lean server API.
                When provided, the server will use Kimina API for verification.
                Example: "http://localhost:14457"
            project_path (str, optional): The local path to a Lean project directory.
                When provided, the server will use local Lean 4 installation.
                Example: "/path/to/lean/project"

        Raises:
            ValueError: If neither `api_url` nor `project_path` is provided.
            
        Note:
            - Server mode requires a running Kimina server at the specified URL
            - Local mode requires Lean 4 and lake to be installed and accessible
            - Local mode takes precedence if both parameters are provided
        """
        if project_path:
            # Prioritize local execution if a project path is provided
            self.mode = "local"
            self.path = project_path
            self.client = None  # Ensures client is not initialized in local mode
            print("LeanServer initialized in LOCAL mode.")
        elif api_url:
            self.mode = "server"
            self.path = api_url
            self.client = KiminaClient(
                api_url=self.path
            )  # The client is initialized once here
            print("LeanServer initialized in SERVER mode.")
        else:
            raise ValueError(
                "You must provide either an 'api_url' for server mode or a 'project_path' for local mode."
            )

    def check_lean_string(self, lean_string: str, add_imports: bool = False):
        """
        Verify a Lean 4 code string using the configured verification method.

        This method checks whether the provided Lean code compiles successfully
        and verifies without errors or 'sorry' statements. It automatically
        handles the verification process based on the server's mode (remote or local).

        Args:
            lean_string (str): The Lean 4 code to be verified.
                Can be a theorem, lemma, definition, or any valid Lean code.
            add_imports (bool, optional): Whether to automatically add standard
                Lean library imports (Mathlib, Aesop, etc.). Defaults to False.

        Returns:
            tuple: A tuple containing three elements:
                - lean_pass (bool): True if the code compiles without errors
                - lean_verify (bool): True if the code compiles AND verifies 
                  without 'sorry' statements or failures
                - output (list or str): Error details if compilation/verification fails,
                  None if successful

        Example:
            >>> lean_server = LeanServer(api_url="http://localhost:14457")
            >>> lean_pass, lean_verify, errors = lean_server.check_lean_string(
            ...     "theorem add_zero (n : ℕ) : n + 0 = n := by simp"
            ... )
            >>> print(f"Compiles: {lean_pass}, Verified: {lean_verify}")
            
        Note:
            - lean_pass=True means the code compiles without syntax/type errors
            - lean_verify=True means the code compiles AND has no 'sorry' statements
            - Both must be True for a fully verified proof
        """
        if self.mode == "server":
            return verify_lean_lemma_server(
                lean_string=lean_string, client=self.client, add_imports=add_imports
            )
        elif self.mode == "local":
            return verify_lean_lemma_local(
                lean_string=lean_string, project_path=self.path, add_imports=add_imports
            )
        else:
            # This case should not be reached due to the __init__ check
            return False, False, "Error: Invalid mode. This is an internal error."
