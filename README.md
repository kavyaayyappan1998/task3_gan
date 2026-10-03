# DATA 266 Lab 1 — Task 3 Standalone

This package contains only Task 3 plus the reproducibility folders required by the lab structure.

## 1. Open in VS Code

Open this `Task3_Standalone` folder as the VS Code workspace.

Install:
- Python extension
- Jupyter extension

Create/activate a virtual environment, then install PyTorch/torchvision for your CPU or CUDA setup
from https://pytorch.org/get-started/locally/

Then run:

```bash
python -m pip install -r requirements.txt
```

## 2. Add the dataset

Copy images into:

```text
task3_gan/data/monet_jpg/
task3_gan/data/photo_jpg/
```

## 3. Run the notebook

Open:

```text
task3_gan/Kavya/src/Part3_CycleGAN_VSCode.ipynb
```

Select the virtual environment as the Jupyter kernel and run cells from top to bottom.

Default full mode:
- 30 epochs
- batch size 1
- image size 256
- 200 images per direction for local metrics

For a quick smoke test, set `DATA266_RUN_MODE=smoke` before starting the VS Code Jupyter kernel.

## 4. Expected outputs

```text
task3_gan/
├── data/
│   ├── monet_jpg/
│   └── photo_jpg/
└── Kavya/
    ├── src/
    │   └── Part3_CycleGAN_VSCode.ipynb
    ├── checkpoints/
    │   └── full/
    │       ├── latest_checkpoint.pt
    │       └── generators_final.pt
    ├── outputs/
    │   ├── pred_A2B/
    │   ├── pred_B2A/
    │   ├── kaggle_pred_A2B/
    │   └── official_evaluation.txt
    ├── metrics/
    ├── check_artifacts.py
    ├── evaluate_local.py          # REPLACE with official Kaggle evaluator
    ├── submission.csv             # ID,FID,MiFID + one result row
    ├── generated_A2B_images.zip   # evidence/backup, not Kaggle score CSV
    ├── full_metrics_report.csv
    ├── human_audit_30.csv
    ├── failure_analysis.md
    ├── results.md
    └── config.json

reproducibility/
├── manifests/
└── raw_logs/
```

## 5. Official Kaggle evaluation and CSV

The class competition requires:

```text
submission.csv
ID,FID,MiFID
1,<official FID>,<official MiFID>
```

Do not use the notebook's 200-image local FID as the competition FID.

1. Download the official competition evaluation script.
2. Replace `task3_gan/Kavya/evaluate_local.py` with that script.
3. Read its usage and set `DATA266_OFFICIAL_EVAL_COMMAND` to its exact command.
4. Rerun the Kaggle inference/evaluation cell.
5. The notebook will capture the evaluator output, parse FID/MiFID, and create the one-row CSV.
6. Before Kaggle submission, set the team name to `PairProgramming_Team_##` using your real team number.

## 6. Local artifact check

After a complete run:

```bash
python task3_gan/Kavya/check_artifacts.py
```

The human audit still requires two people to score the same fixed 30 generated samples.


## Important run notes

- Full mode defaults to 30 epochs. Use epoch 1 timing to estimate whether the full run is practical.
- `latest_checkpoint.pt` is overwritten each epoch for resume.
- Archived full checkpoints are kept every 5 epochs plus the final epoch.
- `generators_final.pt` is the compact generator-only checkpoint for demo/inference.
- Local FID/KID use 200 images per direction in full mode; treat them as local comparison metrics.
- Human audit requires two raters. After ratings are entered, rerun the audit metrics, final metrics,
  and results cells.
- Verify the exact Kaggle class-competition submission protocol before uploading.

## 7. After Kaggle submission

After the score CSV is submitted, set the actual team name and leaderboard rank in the notebook:

```python
os.environ["DATA266_KAGGLE_TEAM_NAME"] = "PairProgramming_Team_XX"
os.environ["DATA266_KAGGLE_RANK"] = "your rank"
```

Then rerun only the `results.md` cell.

The notebook writes official FID/MiFID into both:
- `metrics_report.csv`
- `full_metrics_report.csv`
