"""
run_all_pipeline.py

Reruns the entire demand-forecasting pipeline end to end, in order.
Use this whenever the source workbook changes, a client decision changes
one of the earlier steps, or you just want a clean full rebuild instead of
running each step file by hand.

Stops immediately if any step fails, so you never silently build later
steps on top of a broken earlier one.
"""

import subprocess
import sys
import time

# Update this list if any of your local filenames differ from what's listed
# here - these match what's actually been run so far in this project.
STEPS = [
    "step1_inventory.py",
    "step2_dedupe_inventory.py",
    "step3_clean_tier1.py",
    "step4_recover_group_c_dates.py",
    "step5_count_participants.py",
    "step6_standardize_revenue.py",
    "step7_aggregate_batches.py",
    "step8_feature_engineering.py",
    "step9_train_val_test_split.py",
    "step10_train_models.py",
    "step11_evaluate.py",
    "step12_forecast.py",
]


def run_step(script_name, step_number, total):
    print(f"\n{'='*70}")
    print(f"STEP {step_number}/{total}: {script_name}")
    print("=" * 70)

    start = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ {script_name} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        print("Stopping here - fix this step before continuing. Nothing after it has run.")
        return False

    print(f"\n✅ {script_name} completed in {elapsed:.1f}s")
    return True


def main():
    print(f"Running full pipeline: {len(STEPS)} steps\n")
    overall_start = time.time()

    for i, script in enumerate(STEPS, start=1):
        success = run_step(script, i, len(STEPS))
        if not success:
            sys.exit(1)

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"🎉 ALL {len(STEPS)} STEPS COMPLETED in {total_elapsed:.1f}s")
    print("=" * 70)
    print("\nFinal outputs to check:")
    print("  outputs/forecast.csv           - the 90-day forecast")
    print("  outputs/test_predictions.csv   - model accuracy on held-out data")
    print("  outputs/models/*.joblib        - the retrained models")


if __name__ == "__main__":
    main()