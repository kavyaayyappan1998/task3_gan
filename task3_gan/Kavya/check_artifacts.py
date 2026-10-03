from pathlib import Path
import pandas as pd
import sys

HERE = Path(__file__).resolve().parent
metrics_path = HERE / "full_metrics_report.csv"
submission_path = HERE / "submission.csv"
audit_path = HERE / "human_audit_30.csv"
results_path = HERE / "results.md"
official_output = HERE / "outputs" / "official_evaluation.txt"
a2b = HERE / "outputs" / "pred_A2B"
b2a = HERE / "outputs" / "pred_B2A"
kaggle_a2b = HERE / "outputs" / "kaggle_pred_A2B"

required_metrics = [
    "fid_A2B", "kid_A2B", "generative_precision_A2B", "generative_recall_A2B",
    "cycle_l1_A2B", "lpips_A2B", "content_cosine_A2B",
    "fid_B2A", "kid_B2A", "generative_precision_B2A", "generative_recall_B2A",
    "cycle_l1_B2A", "lpips_B2A", "content_cosine_B2A",
    "final_generator_total_loss", "final_cycle_loss", "final_identity_loss",
    "peak_gradient_norm", "nan_count", "parameter_count_total",
    "training_time_seconds", "images_per_sec", "peak_gpu_memory_mb"
]

errors = []

if not metrics_path.exists():
    errors.append(f"Missing {metrics_path.name}")
else:
    df = pd.read_csv(metrics_path)
    missing = [c for c in required_metrics if c not in df.columns]
    if missing:
        errors.append("Missing metric columns: " + ", ".join(missing))

for required_file in [results_path, audit_path]:
    if not required_file.exists():
        errors.append(f"Missing {required_file.name}")

if not submission_path.exists():
    errors.append("Missing submission.csv")
else:
    sub = pd.read_csv(submission_path)

    if list(sub.columns) != ["ID", "FID", "MiFID"]:
        errors.append(
            "submission.csv columns must be exactly: ID, FID, MiFID"
        )

    if len(sub) != 1:
        errors.append("submission.csv must contain exactly one result row")

    if set(["ID", "FID", "MiFID"]).issubset(sub.columns) and len(sub) == 1:
        if int(sub.iloc[0]["ID"]) != 1:
            errors.append("submission.csv ID must be numeric 1")

        for col in ["FID", "MiFID"]:
            try:
                float(sub.iloc[0][col])
            except Exception:
                errors.append(f"submission.csv {col} must be numeric")

if not official_output.exists():
    errors.append("Missing outputs/official_evaluation.txt")

for folder in [a2b, b2a, kaggle_a2b]:
    if not folder.exists() or not any(folder.glob("*.jpg")):
        errors.append(f"No generated JPGs found in {folder.relative_to(HERE)}")

if errors:
    print("ARTIFACT CHECK: FAILED")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("ARTIFACT CHECK: PASSED")
print("Metrics:", metrics_path)
print("Local A->B images:", len(list(a2b.glob("*.jpg"))))
print("Local B->A images:", len(list(b2a.glob("*.jpg"))))
print("Full Kaggle A->B images:", len(list(kaggle_a2b.glob("*.jpg"))))
print("\nsubmission.csv")
print(pd.read_csv(submission_path))
