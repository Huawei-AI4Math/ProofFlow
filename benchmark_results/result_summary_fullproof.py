import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


def analyze_experiment_results(folder_path: str, output_file: Optional[str] = None) -> Dict:
    """
    Analyze experiment results and calculate various metrics for both legacy and benchmark3 formats.

    Args:
        folder_path: Path to the folder containing JSON files
        output_file: Optional output file name for saving results

    Returns:
        dict: Dictionary containing all statistical information
    """
    folder_path = Path(folder_path)

    # Check if folder exists
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return {}

    # Get all JSON files
    json_files = list(folder_path.glob("*.json"))
    total_files = len(json_files)

    print(f"\n=== PROCESSING FILES ===")
    print(f"Total files: {total_files}")

    if total_files == 0:
        print("No JSON files found")
        return {}

    # Initialize statistics variables
    stats = {
        'total_files': total_files,
        'verified_count': 0,
        'pass_at_1': 0,
        'pass_at_3': 0,
        'pass_at_5': 0,
        'fully_verified_count': 0,
        'verified_files': [],
        'lean_verify_files': []
    }

    # Lists for calculating averages
    total_tokens_list = []
    total_time_list = []
    avg_tokens_per_trial_list = []
    avg_time_per_trial_list = []

    # Pass@k token and time metrics
    pass_at_1_tokens = []
    pass_at_1_times = []
    pass_at_3_tokens = []
    pass_at_3_times = []

    # Retry metrics
    total_retry_3_time = 0.0
    total_retry_3_tokens = 0
    total_retry_1_time = 0.0
    total_retry_1_tokens = 0
    retry_files_count = 0

    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_processed = False

            # Process legacy format (original format)
            if _process_legacy_format(json_file, data, stats, pass_at_1_tokens, pass_at_1_times,
                                      pass_at_3_tokens, pass_at_3_times):
                file_processed = True

            # Process benchmark3 format (step-by-step format)
            if _process_benchmark3_format(json_file, data, stats):
                file_processed = True

            # Calculate retry metrics for both formats
            total_retry_1_time, total_retry_1_tokens, total_retry_3_time, total_retry_3_tokens, retry_files_count = _calculate_retry_metrics(
                data, total_retry_1_time, total_retry_1_tokens, total_retry_3_time, total_retry_3_tokens,
                retry_files_count)

            # Collect general statistics if available
            if "total_tokens" in data:
                total_tokens_list.append(data["total_tokens"])
            if "total_time" in data:
                total_time_list.append(data["total_time"])
            if "avg_tokens_per_trial" in data:
                avg_tokens_per_trial_list.append(data["avg_tokens_per_trial"])
            if "avg_time_per_trial" in data:
                avg_time_per_trial_list.append(data["avg_time_per_trial"])

        except Exception as e:
            print(f"Error processing file {json_file.name}: {e}")

    # Calculate final statistics
    _calculate_final_statistics(stats, total_tokens_list, total_time_list,
                                avg_tokens_per_trial_list, avg_time_per_trial_list,
                                pass_at_1_tokens, pass_at_1_times, pass_at_3_tokens, pass_at_3_times,
                                total_retry_1_time, total_retry_1_tokens, total_retry_3_time,
                                total_retry_3_tokens, retry_files_count)

    # Print results
    _print_comprehensive_results(stats)

    # Save results if output file specified
    if output_file:
        _save_results_to_file(stats, output_file)

    return stats


def _process_legacy_format(json_file: Path, data: Dict, stats: Dict,
                           pass_at_1_tokens: List, pass_at_1_times: List,
                           pass_at_3_tokens: List, pass_at_3_times: List) -> bool:
    """Process files in legacy format (original format)."""
    lean_results = data.get("Lean_results", {})
    if not lean_results:
        return False

    lean_verify = lean_results.get("lean_verify", False)

    if lean_verify:
        stats['verified_count'] += 1
        tries = lean_results.get("tries", 0)

        stats['verified_files'].append((json_file.name, tries))

        # Calculate pass@1 and pass@3
        if tries <= 1:
            stats['pass_at_1'] += 1
        if tries <= 3:
            stats['pass_at_3'] += 1
        if tries <= 5:
            stats['pass_at_5'] += 1

    # Calculate pass@k token and time metrics
    attempt_history = lean_results.get("attempt_history", [])
    if attempt_history:
        # Pass@1: only first attempt
        first_attempt = attempt_history[0]
        if "tokens" in first_attempt:
            pass_at_1_tokens.append(first_attempt["tokens"])
        if "time" in first_attempt:
            pass_at_1_times.append(first_attempt["time"])

        # Pass@3: sum of first three attempts
        total_tokens_3 = sum(attempt.get("tokens", 0) for attempt in attempt_history[:3])
        total_time_3 = sum(attempt.get("time", 0) for attempt in attempt_history[:3])

        if total_tokens_3 > 0:
            pass_at_3_tokens.append(total_tokens_3)
        if total_time_3 > 0:
            pass_at_3_times.append(total_time_3)

    return True


def _process_benchmark3_format(json_file: Path, data: Dict, stats: Dict) -> bool:
    """Process files in benchmark3 format (step-by-step format)."""
    if 'fully_verified' not in data:
        return False

    # Check if theorem is fully verified (this counts as pass@5)
    if data.get('fully_verified') == True:
        stats['fully_verified_count'] += 1
        stats['pass_at_5'] += 1

    # Check for lean_verify and analyze pass@k metrics
    lean_verification = data.get('lean_verification', {})
    if lean_verification.get('lean_verify') == True:
        stats['lean_verify_files'].append(json_file.name)

    return True


def _calculate_retry_metrics(data: Dict, total_retry_1_time: float,
                             total_retry_1_tokens: int, total_retry_3_time: float,
                             total_retry_3_tokens: int, retry_files_count: int) -> Tuple[float, int, float, int, int]:
    """Calculate retry metrics for each problem - always include data regardless of verification status."""
    problem_retry_3_time = 0.0
    problem_retry_3_tokens = 0
    problem_retry_1_time = 0.0
    problem_retry_1_tokens = 0

    # Process theorem_lean_results
    theorem_results = data.get('theorem_lean_results', {})
    attempt_history = theorem_results.get('attempt_history', [])

    for i, attempt in enumerate(attempt_history[:3]):
        if 'time' in attempt:
            problem_retry_3_time += attempt['time']
        if 'tokens' in attempt:
            problem_retry_3_tokens += attempt['tokens']

        if i == 0:  # First attempt only for retry@1
            if 'time' in attempt:
                problem_retry_1_time += attempt['time']
            if 'tokens' in attempt:
                problem_retry_1_tokens += attempt['tokens']

    # Process proof_steps
    proof_steps = data.get('proof_steps', [])
    for step in proof_steps:
        step_lean_results = step.get('lean_results', {})
        step_attempt_history = step_lean_results.get('attempt_history', [])

        for i, attempt in enumerate(step_attempt_history[:3]):
            if 'time' in attempt:
                problem_retry_3_time += attempt['time']
            if 'tokens' in attempt:
                problem_retry_3_tokens += attempt['tokens']

            if i == 0:  # First attempt only for retry@1
                if 'time' in attempt:
                    problem_retry_1_time += attempt['time']
                if 'tokens' in attempt:
                    problem_retry_1_tokens += attempt['tokens']

    # Also process legacy format attempt_history
    legacy_results = data.get("Lean_results", {})
    legacy_attempt_history = legacy_results.get("attempt_history", [])

    for i, attempt in enumerate(legacy_attempt_history[:3]):
        if 'time' in attempt:
            problem_retry_3_time += attempt['time']
        if 'tokens' in attempt:
            problem_retry_3_tokens += attempt['tokens']

        if i == 0:  # First attempt only for retry@1
            if 'time' in attempt:
                problem_retry_1_time += attempt['time']
            if 'tokens' in attempt:
                problem_retry_1_tokens += attempt['tokens']

    return (total_retry_1_time + problem_retry_1_time,
            total_retry_1_tokens + problem_retry_1_tokens,
            total_retry_3_time + problem_retry_3_time,
            total_retry_3_tokens + problem_retry_3_tokens,
            retry_files_count + 1)


def _calculate_final_statistics(stats: Dict, total_tokens_list: List, total_time_list: List,
                                avg_tokens_per_trial_list: List, avg_time_per_trial_list: List,
                                pass_at_1_tokens: List, pass_at_1_times: List,
                                pass_at_3_tokens: List, pass_at_3_times: List,
                                total_retry_1_time: float, total_retry_1_tokens: int,
                                total_retry_3_time: float, total_retry_3_tokens: int,
                                retry_files_count: int) -> None:
    """Calculate final statistics and averages."""

    def safe_average(lst):
        return sum(lst) / len(lst) if lst else 0

    # Basic averages
    stats['avg_total_tokens'] = safe_average(total_tokens_list)
    stats['avg_total_time'] = safe_average(total_time_list)
    stats['avg_tokens_per_trial'] = safe_average(avg_tokens_per_trial_list)
    stats['avg_time_per_trial'] = safe_average(avg_time_per_trial_list)

    # Pass@k averages
    stats['avg_pass_at_1_tokens'] = safe_average(pass_at_1_tokens)
    stats['avg_pass_at_1_time'] = safe_average(pass_at_1_times)
    stats['avg_pass_at_3_tokens'] = safe_average(pass_at_3_tokens)
    stats['avg_pass_at_3_time'] = safe_average(pass_at_3_times)

    # Retry averages
    stats['avg_retry_1_time'] = total_retry_1_time / retry_files_count if retry_files_count > 0 else 0
    stats['avg_retry_1_tokens'] = total_retry_1_tokens / retry_files_count if retry_files_count > 0 else 0
    stats['avg_retry_3_time'] = total_retry_3_time / retry_files_count if retry_files_count > 0 else 0
    stats['avg_retry_3_tokens'] = total_retry_3_tokens / retry_files_count if retry_files_count > 0 else 0

    # Rates
    stats['verification_rate'] = (stats['verified_count'] / stats['total_files'] * 100) if stats[
                                                                                               'total_files'] > 0 else 0
    stats['fully_verified_rate'] = (stats['fully_verified_count'] / stats['total_files'] * 100) if stats[
                                                                                                       'total_files'] > 0 else 0


def _print_comprehensive_results(stats: Dict) -> None:
    """Print comprehensive statistical results in the format matching example.txt."""
    print("=" * 80)
    print("EXPERIMENT ANALYSIS RESULTS")
    print("=" * 80)
    print()

    print("FILE OVERVIEW:")
    print(f"  Total files analyzed: {stats['total_files']}")
    print(f"  Files with complete data: {stats['total_files']}")
    print()

    print("PASS@K METRICS:")
    print(f"  Pass@1: {stats['pass_at_1']} files ({stats['pass_at_1'] / stats['total_files'] * 100:.1f}%)")
    print(f"  Pass@3: {stats['pass_at_3']} files ({stats['pass_at_3'] / stats['total_files'] * 100:.1f}%)")
    print(f"  Pass@5: {stats['pass_at_5']} files ({stats['pass_at_5'] / stats['total_files'] * 100:.1f}%)")
    print()

    print("RETRY METRICS:")
    print(f"  Files with retry data: {stats['total_files']}")
    print(f"  Retry@1 avg time: {stats['avg_retry_1_time']:.1f}s")
    print(f"  Retry@1 avg tokens: {int(stats['avg_retry_1_tokens'])}")
    print(f"  Retry@3 avg time: {stats['avg_retry_3_time']:.1f}s")
    print(f"  Retry@3 avg tokens: {int(stats['avg_retry_3_tokens'])}")
    print(f"  Retry@5 avg time: {stats['avg_total_time']:.1f}s")
    print(f"  Retry@5 avg tokens: {int(stats['avg_total_tokens'])}")
    print()

    if stats['lean_verify_files'] or stats['verified_files']:
        print("LEAN VERIFY FILES:")
        # Print benchmark3 lean_verify files
        for filename in stats['lean_verify_files']:
            print(f"  - {filename}")
        # Print legacy verified files with tries
        for filename, tries in stats['verified_files']:
            print(f"  - {filename}; tries: {tries}")


def _save_results_to_file(stats: Dict, output_file: str) -> None:
    """Save results to TXT file."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write("\n")

        f.write("FILE OVERVIEW:\n")
        f.write(f"  Total files analyzed: {stats['total_files']}\n")
        f.write(f"  Files with complete data: {stats['total_files']}\n")
        f.write("\n")

        f.write("PASS@K METRICS:\n")
        f.write(f"  Pass@1: {stats['pass_at_1']} files ({stats['pass_at_1'] / stats['total_files'] * 100:.1f}%)\n")
        f.write(f"  Pass@3: {stats['pass_at_3']} files ({stats['pass_at_3'] / stats['total_files'] * 100:.1f}%)\n")
        f.write(f"  Pass@5: {stats['pass_at_5']} files ({stats['pass_at_5'] / stats['total_files'] * 100:.1f}%)\n")
        f.write("\n")

        f.write("RETRY METRICS:\n")
        f.write(f"  Files with retry data: {stats['total_files']}\n")
        f.write(f"  Retry@1 avg time: {stats['avg_retry_1_time']:.1f}s\n")
        f.write(f"  Retry@1 avg tokens: {int(stats['avg_retry_1_tokens'])}\n")
        f.write(f"  Retry@3 avg time: {stats['avg_retry_3_time']:.1f}s\n")
        f.write(f"  Retry@3 avg tokens: {int(stats['avg_retry_3_tokens'])}\n")
        f.write(f"  Retry@5 avg time: {stats['avg_total_time']:.1f}s\n")
        f.write(f"  Retry@5 avg tokens: {int(stats['avg_total_tokens'])}\n")
        f.write("\n")

        if stats['lean_verify_files'] or stats['verified_files']:
            f.write("LEAN VERIFY FILES:\n")
            # Write benchmark3 lean_verify files
            for filename in stats['lean_verify_files']:
                f.write(f"  - {filename}\n")
            # Write legacy verified files with tries
            for filename, tries in stats['verified_files']:
                f.write(f"  - {filename}; tries: {tries}\n")


if __name__ == "__main__":
    """Main function to run the analysis."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Script for processing input and output paths')

    # Add arguments
    parser.add_argument('-i', type=str, default="output_pickle/benchmark_fullproof_nothink", help='Input file or directory path (default: output/benchmark2)')
    parser.add_argument('-o', type=str, default="output_tables/fullproof_results_summary.txt", help='Output file or directory path (default: fullproof_results_summary.txt)')


    # Parse arguments
    args = parser.parse_args()

    input_path = args.i
    output_path = args.o

    # Display received parameters
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    # Run analysis
    results = analyze_experiment_results(input_path, output_path)