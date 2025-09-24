import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class ExperimentAnalyzer:
    """
    Unified analyzer for both legacy and benchmark3 experiment result formats.
    """

    def __init__(self, folder_path: str):
        """Initialize the analyzer with folder path."""
        self.folder_path = Path(folder_path)
        self.stats = self._initialize_stats()

    def _initialize_stats(self) -> Dict:
        """Initialize statistics dictionary."""
        return {
            'total_files': 0,
            'verified_count': 0,
            'fully_verified_count': 0,
            'pass_at_1': 0,
            'pass_at_3': 0,
            'pass_at_5': 0,
            'total_verified_steps': 0,
            'total_all_steps': 0,
            'step_pass_at_1_count': 0,
            'step_pass_at_3_count': 0,
            'step_pass_at_5_count': 0,
            'total_steps_analyzed': 0,
            'verified_files': [],
            'lean_verify_files': [],
            'files_with_complete_data': 0,
            'retry_files_count': 0
        }

    def analyze(self) -> Dict:
        """
        Main analysis function that processes all JSON files in the folder.

        Returns:
            Dict containing comprehensive analysis results
        """
        if not self.folder_path.exists():
            print(f"Error: Folder {self.folder_path} does not exist")
            return self.stats

        json_files = list(self.folder_path.glob("*.json"))
        self.stats['total_files'] = len(json_files)

        if len(json_files) == 0:
            print(f"No JSON files found in {self.folder_path}")
            return self.stats

        print(f"Analyzing {len(json_files)} JSON files...")

        # Initialize metric collection lists
        metrics = self._initialize_metrics()

        # Process each file
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self._process_file(json_file, data, metrics)

            except json.JSONDecodeError:
                print(f"Warning: File {json_file.name} is not valid JSON")
            except Exception as e:
                print(f"Warning: Error processing {json_file.name}: {e}")

        # Calculate final statistics
        self._calculate_final_statistics(metrics)

        return self.stats

    def _initialize_metrics(self) -> Dict:
        """Initialize metric collection structures."""
        return {
            'total_tokens_list': [],
            'total_time_list': [],
            'avg_tokens_per_trial_list': [],
            'avg_time_per_trial_list': [],
            'pass_at_1_tokens': [],
            'pass_at_1_times': [],
            'pass_at_3_tokens': [],
            'pass_at_3_times': [],
            'retry_1_time_sum': 0.0,
            'retry_1_tokens_sum': 0,
            'retry_3_time_sum': 0.0,
            'retry_3_tokens_sum': 0
        }

    def _process_file(self, json_file: Path, data: Dict, metrics: Dict) -> None:
        """Process a single JSON file for both legacy and benchmark3 formats."""

        # Try legacy format processing
        if self._process_legacy_format(json_file, data, metrics):
            pass

        # Try benchmark3 format processing
        if self._process_benchmark3_format(json_file, data, metrics):
            pass

        # Collect general performance metrics
        self._collect_general_metrics(data, metrics)

        # Calculate retry metrics
        self._calculate_file_retry_metrics(data, metrics)

    def _process_legacy_format(self, json_file: Path, data: Dict, metrics: Dict) -> bool:
        """Process legacy format files."""
        lean_results = data.get("Lean_results", {})
        if not lean_results:
            return False

        lean_verify = lean_results.get("lean_verify", False)

        if lean_verify:
            self.stats['verified_count'] += 1
            tries = lean_results.get("tries", 0)
            self.stats['verified_files'].append((json_file.name, tries))

            if tries <= 1:
                self.stats['pass_at_1'] += 1
            if tries <= 3:
                self.stats['pass_at_3'] += 1

        # Process attempt history for token/time metrics
        attempt_history = lean_results.get("attempt_history", [])
        if attempt_history:
            self._process_attempt_history(attempt_history, metrics)

        return True

    def _process_benchmark3_format(self, json_file: Path, data: Dict, metrics: Dict) -> bool:
        """Process benchmark3 format files."""
        if 'fully_verified' not in data and 'verified_steps_count' not in data:
            return False

        # Process full verification status
        if data.get('fully_verified') == True:
            self.stats['fully_verified_count'] += 1

        # Process step counts
        verified_steps = data.get('verified_steps_count', 0)
        total_steps = data.get('total_steps_count', 0)

        self.stats['total_verified_steps'] += verified_steps
        self.stats['total_all_steps'] += total_steps

        # Process proof steps for step-level metrics
        self._process_proof_steps(data.get('proof_steps', []))

        # Check lean verification status
        lean_verification = data.get('lean_verification', {})
        if lean_verification.get('lean_verify') == True:
            self.stats['lean_verify_files'].append(json_file.name)
            self._check_pass_at_k_qualification(data)

        return True

    def _process_proof_steps(self, proof_steps: List[Dict]) -> None:
        """Process individual proof steps for step-level metrics."""
        for step in proof_steps:
            lean_results = step.get('lean_results', {})
            lean_pass = lean_results.get('lean_pass', False)
            tries = lean_results.get('tries', float('inf'))

            self.stats['total_steps_analyzed'] += 1

            if lean_pass and tries <= 1:
                self.stats['step_pass_at_1_count'] += 1
            if lean_pass and tries <= 3:
                self.stats['step_pass_at_3_count'] += 1
            if lean_pass and tries <= 5:
                self.stats['step_pass_at_5_count'] += 1

    def _check_pass_at_k_qualification(self, data: Dict) -> None:
        """Check if a file qualifies for pass@k metrics in benchmark3 format."""
        theorem_results = data.get('theorem_lean_results', {})
        theorem_tries = theorem_results.get('tries', float('inf'))

        if theorem_tries <= 5:
            proof_steps = data.get('proof_steps', [])

            all_steps_pass_at_1 = (theorem_tries <= 1 and
                                   all(step.get('tries', float('inf')) <= 1
                                       for step in proof_steps))
            all_steps_pass_at_3 = (theorem_tries <= 3 and
                                   all(step.get('tries', float('inf')) <= 3
                                       for step in proof_steps))
            all_steps_pass_at_5 = all(step.get('tries', float('inf')) <= 5
                                      for step in proof_steps)

            if all_steps_pass_at_1:
                self.stats['pass_at_1'] += 1
            if all_steps_pass_at_3:
                self.stats['pass_at_3'] += 1
            if all_steps_pass_at_5:
                self.stats['pass_at_5'] += 1

    def _process_attempt_history(self, attempt_history: List[Dict], metrics: Dict) -> None:
        """Process attempt history for token and time metrics."""
        if not attempt_history:
            return

        # Pass@1: first attempt only
        first_attempt = attempt_history[0]
        if "tokens" in first_attempt:
            metrics['pass_at_1_tokens'].append(first_attempt["tokens"])
        if "time" in first_attempt:
            metrics['pass_at_1_times'].append(first_attempt["time"])

        # Pass@3: sum of first three attempts
        total_tokens_3 = sum(attempt.get("tokens", 0) for attempt in attempt_history[:3])
        total_time_3 = sum(attempt.get("time", 0) for attempt in attempt_history[:3])

        if total_tokens_3 > 0:
            metrics['pass_at_3_tokens'].append(total_tokens_3)
        if total_time_3 > 0:
            metrics['pass_at_3_times'].append(total_time_3)

    def _collect_general_metrics(self, data: Dict, metrics: Dict) -> None:
        """Collect general performance metrics."""
        if "total_tokens" in data:
            metrics['total_tokens_list'].append(data["total_tokens"])
        if "total_time" in data:
            metrics['total_time_list'].append(data["total_time"])
        if "avg_tokens_per_trial" in data:
            metrics['avg_tokens_per_trial_list'].append(data["avg_tokens_per_trial"])
        if "avg_time_per_trial" in data:
            metrics['avg_time_per_trial_list'].append(data["avg_time_per_trial"])

        if "total_tokens" in data and "total_time" in data:
            self.stats['files_with_complete_data'] += 1

    def _calculate_file_retry_metrics(self, data: Dict, metrics: Dict) -> None:
        """Calculate retry metrics for a single file."""
        file_retry_1_time = 0.0
        file_retry_1_tokens = 0
        file_retry_3_time = 0.0
        file_retry_3_tokens = 0

        # Process theorem results
        theorem_results = data.get('theorem_lean_results', {})
        theorem_attempts = theorem_results.get('attempt_history', [])

        for i, attempt in enumerate(theorem_attempts[:3]):
            if 'time' in attempt:
                file_retry_3_time += attempt['time']
                if i == 0:
                    file_retry_1_time += attempt['time']
            if 'tokens' in attempt:
                file_retry_3_tokens += attempt['tokens']
                if i == 0:
                    file_retry_1_tokens += attempt['tokens']

        # Process proof steps
        proof_steps = data.get('proof_steps', [])
        for step in proof_steps:
            step_attempts = step.get('attempt_history', [])

            for i, attempt in enumerate(step_attempts[:3]):
                if 'time' in attempt:
                    file_retry_3_time += attempt['time']
                    if i == 0:
                        file_retry_1_time += attempt['time']
                if 'tokens' in attempt:
                    file_retry_3_tokens += attempt['tokens']
                    if i == 0:
                        file_retry_1_tokens += attempt['tokens']

        # Add to totals
        metrics['retry_1_time_sum'] += file_retry_1_time
        metrics['retry_1_tokens_sum'] += file_retry_1_tokens
        metrics['retry_3_time_sum'] += file_retry_3_time
        metrics['retry_3_tokens_sum'] += file_retry_3_tokens

        if file_retry_1_time > 0 or file_retry_1_tokens > 0:
            self.stats['retry_files_count'] += 1

    def _calculate_final_statistics(self, metrics: Dict) -> None:
        """Calculate final statistics and averages."""

        def safe_average(lst):
            return sum(lst) / len(lst) if lst else 0

        # Basic performance averages
        self.stats['avg_total_tokens'] = safe_average(metrics['total_tokens_list'])
        self.stats['avg_total_time'] = safe_average(metrics['total_time_list'])
        self.stats['avg_tokens_per_trial'] = safe_average(metrics['avg_tokens_per_trial_list'])
        self.stats['avg_time_per_trial'] = safe_average(metrics['avg_time_per_trial_list'])

        # Pass@k averages
        self.stats['avg_pass_at_1_tokens'] = safe_average(metrics['pass_at_1_tokens'])
        self.stats['avg_pass_at_1_time'] = safe_average(metrics['pass_at_1_times'])
        self.stats['avg_pass_at_3_tokens'] = safe_average(metrics['pass_at_3_tokens'])
        self.stats['avg_pass_at_3_time'] = safe_average(metrics['pass_at_3_times'])

        # Retry averages
        retry_count = self.stats['retry_files_count']
        self.stats['avg_retry_1_time'] = metrics['retry_1_time_sum'] / retry_count if retry_count > 0 else 0
        self.stats['avg_retry_1_tokens'] = metrics['retry_1_tokens_sum'] / retry_count if retry_count > 0 else 0
        self.stats['avg_retry_3_time'] = metrics['retry_3_time_sum'] / retry_count if retry_count > 0 else 0
        self.stats['avg_retry_3_tokens'] = metrics['retry_3_tokens_sum'] / retry_count if retry_count > 0 else 0

        # Success rates
        total = self.stats['total_files']
        self.stats['verification_rate'] = (self.stats['verified_count'] / total * 100) if total > 0 else 0
        self.stats['fully_verified_rate'] = (self.stats['fully_verified_count'] / total * 100) if total > 0 else 0
        self.stats['pass_at_1_rate'] = (self.stats['pass_at_1'] / total * 100) if total > 0 else 0
        self.stats['pass_at_3_rate'] = (self.stats['pass_at_3'] / total * 100) if total > 0 else 0

        # Step-level rates
        total_steps = self.stats['total_steps_analyzed']
        self.stats['step_pass_at_1_rate'] = (
                    self.stats['step_pass_at_1_count'] / total_steps * 100) if total_steps > 0 else 0
        self.stats['step_pass_at_3_rate'] = (
                    self.stats['step_pass_at_3_count'] / total_steps * 100) if total_steps > 0 else 0
        self.stats['step_pass_at_5_rate'] = (
                    self.stats['step_pass_at_5_count'] / total_steps * 100) if total_steps > 0 else 0
        self.stats['step_success_rate'] = (self.stats['total_verified_steps'] / self.stats['total_all_steps'] * 100) if \
        self.stats['total_all_steps'] > 0 else 0

        # Calculate Pass@5 rate (same as fully_verified_rate)
        self.stats['pass_at_5_rate'] = self.stats['fully_verified_rate']

    def print_and_save_results(self, output_file: str = None) -> None:
        """Print comprehensive analysis results and optionally save to file."""
        # Capture the output in a string
        output_lines = []

        output_lines.append("=" * 80)
        output_lines.append("EXPERIMENT ANALYSIS RESULTS")
        output_lines.append("=" * 80)

        output_lines.append(f"\nFILE OVERVIEW:")
        output_lines.append(f"  Total files analyzed: {self.stats['total_files']}")
        output_lines.append(f"  Files with complete data: {self.stats['files_with_complete_data']}")

        output_lines.append(f"\nPASS@K METRICS:")
        output_lines.append(f"  Pass@1: {self.stats['pass_at_1']} files ({self.stats['pass_at_1_rate']:.1f}%)")
        output_lines.append(f"  Pass@3: {self.stats['pass_at_3']} files ({self.stats['pass_at_3_rate']:.1f}%)")
        output_lines.append(
            f"  Pass@5: {self.stats['fully_verified_count']} files ({self.stats['fully_verified_rate']:.1f}%)")

        if self.stats['total_steps_analyzed'] > 0:
            output_lines.append(f"\nSTEP-LEVEL ANALYSIS:")
            output_lines.append(f"  Total steps analyzed: {self.stats['total_steps_analyzed']}")
            output_lines.append(
                f"  Step Pass@1: {self.stats['step_pass_at_1_count']} ({self.stats['step_pass_at_1_rate']:.1f}%)")
            output_lines.append(
                f"  Step Pass@3: {self.stats['step_pass_at_3_count']} ({self.stats['step_pass_at_3_rate']:.1f}%)")
            output_lines.append(
                f"  Step Pass@5: {self.stats['total_verified_steps']} ({self.stats['step_success_rate']:.1f}%)")

        if self.stats['retry_files_count'] > 0:
            output_lines.append(f"\nRETRY METRICS:")
            output_lines.append(f"  Files with retry data: {self.stats['retry_files_count']}")
            output_lines.append(f"  Retry@1 avg time: {self.stats['avg_retry_1_time']:.1f}s")
            output_lines.append(f"  Retry@1 avg tokens: {self.stats['avg_retry_1_tokens']:.0f}")
            output_lines.append(f"  Retry@3 avg time: {self.stats['avg_retry_3_time']:.1f}s")
            output_lines.append(f"  Retry@3 avg tokens: {self.stats['avg_retry_3_tokens']:.0f}")
            output_lines.append(f"  Retry@5 avg time: {self.stats['avg_total_time']:.1f}s")
            output_lines.append(f"  Retry@5 avg tokens: {self.stats['avg_total_tokens']:.0f}")

        if self.stats['lean_verify_files']:
            output_lines.append(f"\nLEAN VERIFY FILES:")
            for filename in self.stats['lean_verify_files']:
                output_lines.append(f"  - {filename}")

        # Print to console
        for line in output_lines:
            print(line)

        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(output_lines))
            except Exception as e:
                print(f"Error saving results: {e}")

    def save_results(self, output_file: str) -> None:
        """Deprecated - use print_and_save_results instead."""
        pass

    def generate_report(self, report_file: str) -> None:
        """Deprecated - use print_and_save_results instead."""
        pass


def analyze_experiments(folder_path: str, output_txt: str = "analysis_results.txt") -> Dict:
    """
    Convenience function to run complete analysis.

    Args:
        folder_path: Path to folder containing JSON experiment files
        output_txt: Output text file name

    Returns:
        Dictionary containing analysis results
    """
    analyzer = ExperimentAnalyzer(folder_path)
    results = analyzer.analyze()

    analyzer.print_and_save_results(output_txt)

    return results


if __name__ == "__main__":
    """Main function to run the analysis."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Script for processing input and output paths')

    # Add arguments
    parser.add_argument('-i', type=str, default="output_pickle/benchmark_stepproof_nothink", help='Input file or directory path (default: output/benchmark2)')
    parser.add_argument('-o', type=str, default="output_tables/stepproof_results_summary.txt", help='Output file or directory path (default: fullproof_results_summary.txt)')


    # Parse arguments
    args = parser.parse_args()

    input_path = args.i
    output_path = args.o

    # Display received parameters
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    # Run analysis
    results = analyze_experiments(input_path, output_path)