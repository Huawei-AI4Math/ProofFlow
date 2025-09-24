import json
import os

from dotenv import load_dotenv

from proofflow import LeanServer, LLMManager, ProofFlow, start_vllm_server

load_dotenv()

# Lean Server
lean_server = LeanServer(api_url=os.getenv("LEAN_SERVER_URL"))

# -------------------- Initialize LLM Models ----------------------------

# Model for proof graph generation
graph_model = LLMManager(
    model_info={
        "api_key": os.getenv("API_KEY"),
        "base_url": os.getenv("OPEN_AI_BASE_URL"),
        "model": "google/gemini-2.5-pro",
    },
    system_prompt_path="prompts/proof_graph_no_DAG.md",
)

# Model for formalizing lemmas
formalize_model = LLMManager(
    model_info={
        "api_key": os.getenv("API_KEY"),
        "base_url": os.getenv("OPEN_AI_BASE_URL"),
        "model": "google/gemini-2.5-flash",
    },
    system_prompt_path="prompts/lemma_formalizer_no_think.md",
)

# Model for solving proofs
solver_model = LLMManager(
    model_info={
        "api_key": os.getenv("API_KEY"),
        "base_url": os.getenv("OPEN_AI_BASE_URL"),
        "model": "deepseek/deepseek-prover-v2",
    },
    system_prompt_path="prompts/lemma_prover_no_think.md",
)

# Model for solving proofs
score_model = LLMManager(
    model_info={
        "api_key": os.getenv("API_KEY"),
        "base_url": os.getenv("OPEN_AI_BASE_URL"),
        "model": "anthropic/claude-sonnet-4",
    },
    system_prompt_path=None,
)

# Load the JSON file
with open("data/benchmark_0409.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# output directory
folder = "benchmark_results/output_pickle/benckmark5 - no think noDAG"
os.makedirs(folder, exist_ok=True)

for item in reversed(data):
    nl_string = (
        f"Theorem: {item['nl_theorem'].strip()}\n\nProof: {item['nl_proof'].strip()}\n"
    )
    save_name = item["origin"].split(".")[0] + "_" + str(item["id"]) + ".json"

    if os.path.exists(f"{folder}/{save_name}.html"):
        print(f"Skipping {save_name}: file already exists")
        continue

    try:
        formalizer = ProofFlow(
            lean_server=lean_server,
            graph_model_manager=graph_model,
            formalize_model_manager=formalize_model,
            solver_model_manager=solver_model,
        )
        formalizer.autoformalize_series(
            nl_string, formalizer_retries=5, prover_retries=5, follow_dag=False
        )
        formalizer.proof_score()
        formalizer.error_analysis(prover_retries=3)
        formalizer.interactive_dag(filepath=f"{folder}/{save_name}.html")
        formalizer.save(filepath=f"{folder}/{save_name}.pickle")

    except Exception as e:
        print(f"ERROR {save_name}: {str(e)}")
