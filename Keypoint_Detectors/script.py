import subprocess
import sys

"""
	This script runs the feature extraction, evaluation, and keypoint matching file.
	It is designed to be run from the command line and will execute each script in order,
	and stream directly to the terminal.

	TO RUN:
	python script.py
"""

# This function is used to run file code.
def run_code(script, args=None):
	cmd = [sys.executable, script]
	if args:
		cmd += args
	print(f"\nRunning: {script}")
	subprocess.run(cmd)  # This will stream all output directly to your terminal


if __name__ == "__main__":
	print("=== SUMMARY ===")

	run_code("feature_extraction.py")

	run_code("evaluate_sift.py")

	run_code("evaluate_hog.py")

	run_code("keypoint_matching.py")

	print("\n=== Script completed successfully. ===")