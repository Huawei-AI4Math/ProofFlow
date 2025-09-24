import os
import re
import sys
import traceback
import inspect
import pandas as pd

from proofflow import LeanServer, LLMManager, ProofFlow
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

formalizer = ProofFlow(lean_server=None,
                       graph_model_manager=None,
                       formalize_model_manager=None,
                       solver_model_manager=None)

def process_run(folder, pass_at):
    """
    Processes a directory of pickle files, extracting and summarizing
    formalization and solving metrics.

    Args:
        folder (str): The path to the directory containing pickle files.
        pass_at (int): The pass_at value to use for formalization summary.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data and an average row.
    """
    # ... (the loop to create the data list remains the same)
    files = [f for f in os.listdir(folder) if f.endswith(('.pickle', '.pkl'))]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
    
    data = []
    # Note: 'formalizer' would be instantiated here in a real scenario
    
    for f in files:
        filepath = os.path.join(folder, f)
        formalizer.load(filepath=filepath)
        
        s = formalizer.summary(verbose=False, pass_at=pass_at)

        row = {
            'file': os.path.splitext(f)[0],
            'nl_proof': ILLEGAL_CHARACTERS_RE.sub('', formalizer.nl_proof),
            'form_acc': s['form_acc'],
            'solv_acc': s['solv_acc'],
            'total_time': sum([t["elapsed_time"] for t in formalizer.elapsed_time(verbose=False) if not t["prove_negation"]]),
            'generated_tokens': s['generated_tokens'],
            'total_llm_calls': s['total_calls'],
            'lean_code': ILLEGAL_CHARACTERS_RE.sub('', formalizer.get_lean_code()),
            'total_score': float(formalizer.total_score(pass_at, aggregation="equal")),
            'correct_syntax': 1 if s['solv_acc'] == 1 else 0
        }
        data.append(row)

    df = pd.DataFrame(data)


    # Compute averages for numeric columns
    numeric_cols = ['form_acc', 'solv_acc', 'total_time', 'generated_tokens', 'total_llm_calls', 'total_score', 'correct_syntax']
    averages = df[numeric_cols].mean(numeric_only=True).to_dict()

    # Create a new row for the averages
    avg_row = {
        'file': 'Average',
        'nl_proof': None,
        'lean_code': None,
        **averages
    }

    # Ensure file column is string
    df['file'] = df['file'].astype(str)

    # Reorder columns for better readability, including the new column
    cols = ['file', 'nl_proof', 'form_acc', 'solv_acc','total_score', 'correct_syntax',
            'total_time', 'generated_tokens', 'total_llm_calls', 'lean_code']
    df = df[cols]

    # Append the average row to the DataFrame
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    return df

def main():
    """
    Main function to parse command-line arguments and run the processing.
    """
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <folder> <pass_at> <output_file>")
        sys.exit(1)

    folder = sys.argv[1]
    pass_at = int(sys.argv[2])
    output_file = sys.argv[3]

    print(f"Processing folder: {folder}")
    print(f"Using pass_at: {pass_at}")
    print(f"Saving output to: {output_file}")
    
    df = process_run(folder, pass_at)
    # Ensure 'file' column is a string
    df['file'] = df['file'].astype(str)

    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Successfully saved data to {output_file}")
        


if __name__ == "__main__":
    main()