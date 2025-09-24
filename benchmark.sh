#!/bin/bash

# --- Generate .pickle and .html files with autoformalization results ---
# Results saved in output folder
python -m benchmark_results.benchmark_think_noDAG.py
python -m benchmark_results.benchmark_think_DAG.py
python -m benchmark_results.benchmark_no_think_noDAG.py
python -m benchmark_results.benchmark_no_think_DAG.py


# --- Summary table of Benchmark 1: no think DAG (pass@5) ---
python -m benchmark_results.process_benchmark_files "benchmark_results/output_pickle/benckmark1 - no think DAG" 5 "benchmark_results/output_tables/benckmark1 - no think DAG - pass 5.xlsx"

# --- Summary table of Benchmark 3: think DAG (pass@5) ---
python -m benchmark_results.process_benchmark_files "benchmark_results/output_pickle/benckmark3 - think DAG" 5 "benchmark_results/output_tables/benckmark3 - think DAG - pass 5.xlsx"

# --- Summary table of Benchmark 5: no think noDAG (pass@5) ---
python -m benchmark_results.process_benchmark_files "benchmark_results/output_pickle/benckmark5 - no think noDAG" 5 "benchmark_results/output_tables/benckmark5 - no think noDAG - pass 5.xlsx"

# --- Summary table of Benchmark 6: think noDAG (pass@5) ---
python -m benchmark_results.process_benchmark_files "benchmark_results/output_pickle/benckmark6 - think noDAG" 5 "benchmark_results/output_tables/benckmark6 - think noDAG - pass 5.xlsx"




# --- Error analysis results ---
# Results saved to benchmark_results/output_tables/error_analysis.xlsx
python -m benchmark_results.process_benchmark_error_analysis




# --- Existing Methods ---
# --- Generate .json files with autoformalization results ---
# Results saved in output folder
python -m benchmark_results.full_proof_nothink #--dry-run
python -m benchmark_results.full_proof_think #--dry-run
python -m benchmark_results.step_proof_nothink #--dry-run
python -m benchmark_results.step_proof_think #--dry-run

# --- Summary Results ---
python -m benchmark_results.result_summary_fullproof -i benchmark_results/output_pickle/benchmark_fullproof_nothink -o benchmark_results/output_tables/fullproof_results_summary_nothink.txt
python -m benchmark_results.result_summary_fullproof -i benchmark_results/output_pickle/benchmark_fullproof_think -o benchmark_results/output_tables/fullproof_results_summary_think.txt
python -m benchmark_results.result_summary_stepproof -i benchmark_results/output_pickle/benchmark_stepproof_nothink -o benchmark_results/output_tables/stepproof_results_summary_nothink.txt
python -m benchmark_results.result_summary_stepproof -i benchmark_results/output_pickle/benchmark_stepproof_think -o benchmark_results/output_tables/stepproof_results_summary_think.txt
