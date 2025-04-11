# run_all_experiments.py

import os
import subprocess

# List of experiment IDs to run
EXPERIMENT_IDS = [2, 3]

# Path to main script
MAIN_SCRIPT = "LM/part_A/main.py"

# Loop through experiments and run each one
for exp_id in EXPERIMENT_IDS:
    print(f"\n=== Running Experiment ID: {exp_id} ===")
    subprocess.run(["python", MAIN_SCRIPT, str(exp_id)], check=True)
    print(f"=== Finished Experiment ID: {exp_id} ===\n")
