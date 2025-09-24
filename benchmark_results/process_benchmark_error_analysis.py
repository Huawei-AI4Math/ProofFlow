import os
import pandas as pd
import re
from collections import Counter
from proofflow import ProofFlow

formalizer = ProofFlow(lean_server=None,
                            graph_model_manager=None,
                            formalize_model_manager=None,
                            solver_model_manager=None)

# The `process_run` function provided by the user
def process_run(folder):
    files = [f for f in os.listdir(folder) if f.endswith(('.pickle', '.pkl'))]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
    
    error_types = []
    for f in files:
        filepath = os.path.join(folder, f)
    
        formalizer.load(filepath=filepath)

        for item in formalizer.proof_items:
            et = item.error_report["error_type"]
            if et == "Formalization":
                et = "Formalizer"
            error_types.append(et)

    return Counter(error_types)

## Final Table Generation

data = []
pipelines = ["benchmark_results/output_pickle/benckmark1 - no think DAG", 
             "benchmark_results/output_pickle/benckmark3 - think DAG",
             "benchmark_results/output_pickle/benckmark5 - no think noDAG",
             "benchmark_results/output_pickle/benckmark6 - think noDAG"]

for pipeline_name in pipelines:
    # Get the Counter object for the current pipeline
    error_counts = process_run(pipeline_name)
    
    # Calculate total steps
    total_steps = sum(error_counts.values())
    
    # Initialize row data for the current pipeline
    row = {'pipeline': pipeline_name, 'total_steps': total_steps}
    
    # Calculate and add percentages for each error type
    for error_type in error_counts:
        count = error_counts.get(error_type, 0)
        percentage = (count / total_steps) * 100 if total_steps > 0 else 0
        row[f'{error_type} (%)'] = f"{percentage:.2f}%"
        
    data.append(row)

# Create the final DataFrame
df = pd.DataFrame(data)
df.to_excel("benchmark_results/output_tables/error_analysis.xlsx", index=False)