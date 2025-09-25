# ProofFlow

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ProofFlow is a  Python package that automatically converts natural language mathematical proofs into formalized Lean 4 code using Large Language Models (LLMs). This package is designed to bridge the gap between informal mathematical reasoning and formal verification systems, making mathematical proofs machine-verifiable and accessible to automated theorem provers.

## 🚀 Features

- **Intelligent Proof Graph Generation**: Automatically decomposes natural language proofs into structured dependency graphs
- **Multi-Model Support**: Compatible with various LLMs including Claude, GPT, Gemini, and custom vLLM servers
- **Lean 4 Integration**: Generates valid Lean 4 code with automatic verification
- **Interactive Visualizations**: Create both static and interactive proof dependency graphs
- **Error Analysis**: Comprehensive error detection and analysis for debugging formalizations
- **Semantic Scoring**: Evaluate the quality of formalized proofs using AI-powered scoring
- **Flexible Architecture**: Support for both DAG-based and sequential proof processing

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Lean 4 (for local verification) or access to a Lean server

### Install from Source

```bash
git clone https://github.com/your-username/proofflow.git
cd proofflow
pip install -e .
```

## 🏗️ Project Structure

```
proofflow/
├── proofflow/                  # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── proofflow.py            # Core ProofFlow class
│   ├── proof_graph.py         # Proof graph generation and validation
│   ├── proof_formalize.py     # Natural language to Lean formalization
│   ├── proof_prover.py        # Automated proof generation
│   ├── proof_scorer.py        # Semantic scoring of formalized proofs
│   ├── lean_check.py          # Lean 4 verification utilities
│   ├── utils.py               # Utility functions and LLM management
│   ├── io.py                  # Input/output operations
│   └── vis.py                 # Visualization utilities
├── prompts/                   # LLM prompt templates
│   ├── proof_graph.md         # Proof graph generation prompts
│   ├── lemma_formalizer.md    # Formalization prompts
│   ├── lemma_prover.md        # Proof generation prompts
│   └── ...
├── data/                      # Sample datasets
│   └── benchmark_0409.json    # Benchmark dataset
├── benchmark_results/         # Benchmark results and outputs
├── example.ipynb             # Comprehensive usage examples
├── requirements.txt          # Python dependencies
├── setup.py                  # Package configuration
└── README.md                 # This file
```

## 🚀 Quick Start

### Basic Usage

```python
from proofflow import ProofFlow, LLMManager, LeanServer

# Set up Lean server (local or remote)
lean_server = LeanServer(api_url="http://localhost:14457")  # Remote server
# OR
# lean_server = LeanServer(project_path="/path/to/mathlib")  # Local project

# Configure LLM models
graph_model = LLMManager(
    model_info={
        "api_key": "your-api-key",
        "base_url": "your-base-url",
        "model": "your-model-name",
    },
    system_prompt_path="prompts/proof_graph.md",
)

formalize_model = LLMManager(
    model_info={
        "api_key": "your-api-key",
        "base_url": "your-base-url",
        "model": "your-model-name",
    },
    system_prompt_path="prompts/lemma_formalizer.md",
)

solver_model = LLMManager(
    model_info={
        "api_key": "your-api-key",
        "base_url": "your-base-url",
        "model": "your-model-name",
    },
    system_prompt_path="prompts/lemma_prover.md",
)

# Initialize ProofFlow
proof_flow = ProofFlow(
    lean_server=lean_server,
    graph_model_manager=graph_model,
    formalize_model_manager=formalize_model,
    solver_model_manager=solver_model,
    verbose=True
)

# Process a natural language proof
nl_proof = """
Theorem: For all real numbers x, y, if x² + y² = 1, then |x| ≤ 1.
Proof: Since x² ≥ 0 and y² ≥ 0, we have x² + y² ≥ x². 
Given that x² + y² = 1, we get 1 ≥ x², which means x² ≤ 1. 
Taking the square root of both sides, we obtain |x| ≤ 1.
"""

# Run formalization
proof_flow.autoformalize_series(nl_proof)

# Get results
lean_code = proof_flow.get_lean_code()
print(lean_code)

# Generate visualizations
proof_flow.plot_dag("proof_dag.png")
proof_flow.interactive_dag("proof_dag.html")

# Get performance summary
summary = proof_flow.summary()
print(f"Formalization accuracy: {summary['form_acc']:.2%}")
print(f"Proof success rate: {summary['solv_acc']:.2%}")
```

## 🔧 Configuration

### LLM Models

AutoFormalize supports multiple LLM providers:

- **OpenRouter**: Claude, GPT, Gemini models
- **OpenAI**: GPT-3.5, GPT-4, GPT-4o
- **Custom vLLM servers**: Local model hosting
- **Hugging Face**: Transformers models

### Lean 4 Setup

Choose between local or server-based Lean verification:

```python
# Local Lean project (requires mathlib)
lean_server = LeanServer(project_path="/path/to/your/lean/project")

# Remote Lean server
lean_server = LeanServer(api_url="http://your-lean-server:port")
```

## 📊 Advanced Features

### Semantic Scoring

```python
# Add scoring model
score_model = LLMManager(
    model_info={
        "api_key": "your-api-key",
        "base_url": "your-base-url",
        "model": "your-model-name",
    },
    system_prompt_path=None,
)

proof_flow = ProofFlow(
    # ... other parameters
    score_model_manager=score_model
)

# Compute proof scores after autoformalize series
proof_flow.proof_score(aggregation="katz", verbose=True)
print(f"Total proof score: {proof_flow.total_score}")
```

### Error Analysis

```python
# Perform comprehensive error analysis
proof_flow.error_analysis(
    score_threshold=0.6,
    prover_retries=3,
    verbose=True
)

# Access error reports for each proof step
for item in proof_flow.proof_items:
    if hasattr(item, 'error_report'):
        print(f"Step {item.id}: {item.error_report['error_type']}")
```


### Visualization

```python
# Create static proof graph
proof_flow.plot_dag("proof_structure.png")

# Create interactive HTML visualization
proof_flow.interactive_dag("interactive_proof.html")
```

## 🧪 Benchmarking and Reproducibility

To benchmark the Goedel Formalizer and Solver, run the benchmark.sh script. This script automates the entire process, but it's very time-consuming and can take several days to complete.

Before you start, you'll need to fill out the .env file with the necessary API keys and URLs for your services. This includes the OpenAI API, as well as the Goedel Formalizer and Solver (including their model locations), and the Lean server. An example .env file is left on the main folder, with the API key fields left blank for you to fill in.

Due to potential connection timeouts and rate limits with some services, it's a good idea to run the commands in benchmark.sh one by one in your terminal. If these issues occurs, just retry it to repeat the missing problems. Once the script finishes, the results—including autoformalization files (.pickle, .html) and summary tables (.xlsx)—will be stored in the benchmark_results/ folder.

## 📈 Performance Metrics

AutoFormalize tracks detailed performance metrics:

- **Formalization Accuracy**: Percentage of successfully formalized proof steps
- **Proof Success Rate**: Percentage of steps that can be automatically proven
- **ProofScore**: A novel composite score for the whole proof formalization, taking into account semanting similarity between natural language and Lean code
- **Token Usage**: Total tokens consumed across all LLM calls
- **Processing Time**: Time breakdown by model and operation


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**ProofFlow** - Bridging the gap between informal mathematics and formal verification.