"""Builds launch_eval.ipynb: upload the eval inputs to a private Hub dataset and run run_eval.py on Hugging Face Jobs."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Run the step-3 evaluation on Hugging Face Jobs

This notebook needs **no GPU**. It uploads the inputs the evaluation reads (eval items, chat template, every adapter and the reason-first checkpoints) to a private Hub dataset `<your-username>/reasons-vs-refusals`, then starts `run_eval.py` on a Hugging Face A100 80GB ($2.50/hour, billed per minute). `run_eval.py` is generated from `reasons_eval.ipynb` (same code; Drive replaced by the Hub repo). Each stage's outputs are pushed to the repo as soon as it finishes; if the job stops, starting it again resumes from the last finished stage.

**Before running:** put `run_eval.py` in your `Reasons_vs_Refusals` Drive folder. Make sure your HF token is a *write* token and that the account has Jobs credits.

**Cost (this round):** the four old arms are already evaluated in `eval/` and are loaded, not re-run. New work: the four reason-first arms (generation + judge), representation v2 for all 9 models (decision-point score, prompt-only ablation), and the underperformance audit (15 models × 600 questions + judge). Smoke ≈ 40 min (≈ $1.7); full ≈ 2.5–3 h (≈ $6.5–7.5). The timeout caps the spend; a stopped job resumes from the last saved stage.
""")

code(r"""
ROOT = "/content/drive/MyDrive/AGI_Learning/Lexsi.ai Hackathon/Reasons_vs_Refusals"
ARMS = ["refusal", "reasons", "filler", "shuffled", "reason_first", "lead_first", "reason_first_v2", "lead_first_v2"]
NEW_ARMS = ["reason_first", "lead_first", "reason_first_v2", "lead_first_v2"]
!pip install -q -U huggingface_hub
import os, shutil
from google.colab import drive, userdata
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from huggingface_hub import HfApi, login
login(token=os.environ["HF_TOKEN"])
api = HfApi(); user = api.whoami()["name"]; REPO = f"{user}/reasons-vs-refusals"
for p in ["run_eval.py", "data/eval_items.jsonl", "train/chat_template.jinja"] + [f"train/{a}/final/adapter_model.safetensors" for a in ARMS]:
    assert os.path.exists(f"{ROOT}/{p}"), f"missing {ROOT}/{p}"
print("Hub user:", user, "| repo:", REPO)
""")

code(r"""
# upload the inputs (private dataset repo); re-running only re-uploads changed files
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=f"{ROOT}/data/eval_items.jsonl", path_in_repo="data/eval_items.jsonl", repo_id=REPO, repo_type="dataset")
api.upload_file(path_or_fileobj=f"{ROOT}/train/chat_template.jinja", path_in_repo="train/chat_template.jinja", repo_id=REPO, repo_type="dataset")
for a in ARMS:
    api.upload_folder(folder_path=f"{ROOT}/train/{a}/final", path_in_repo=f"train/{a}/final", repo_id=REPO, repo_type="dataset")
    print("uploaded adapter:", a)
for a in ["reason_first_v2", "lead_first_v2"]:   # intermediate checkpoints for the underperformance audit
    for c in sorted(x for x in os.listdir(f"{ROOT}/train/{a}") if x.startswith("ckpt-")):
        api.upload_folder(folder_path=f"{ROOT}/train/{a}/{c}", path_in_repo=f"train/{a}/{c}", repo_id=REPO, repo_type="dataset")
        print("uploaded checkpoint:", a, c)
print(sorted(f for f in api.list_repo_files(REPO, repo_type="dataset") if not f.endswith(".gitattributes"))[:20])
""")

md(r"""
## Start the job

Run the **smoke** job first (`SMOKE=1`; it adds the new arms to `eval_smoke/`). When it finishes and the outputs look right (next section), run the **full** job (`SMOKE=0`). Logs stream into the cell; stopping the cell does **not** stop the job.
""")

code(r"""
shutil.copy(f"{ROOT}/run_eval.py", "/content/run_eval.py")
SMOKE = 1          # set to 0 for the full run
!hf jobs uv run --flavor a100-large --timeout {"1h" if SMOKE else "4h"} --secrets HF_TOKEN -e SMOKE={SMOKE} /content/run_eval.py
""")

md(r"""
**Useful commands** (in a cell with `!` in front):
- `hf jobs ps -a` — list your jobs and their status
- `hf jobs logs <job_id> | tail -n 60` — latest log lines
- `hf jobs cancel <job_id>` — stop a job (billing stops)
""")

code(r"""
# download the results to Drive (eval_smoke/ or eval/) and send me summary_eval.json + the job log
from huggingface_hub import snapshot_download
sub = "eval_smoke" if SMOKE else "eval"
snapshot_download(REPO, repo_type="dataset", local_dir="/content/hub", allow_patterns=[f"{sub}/*"])
os.makedirs(f"{ROOT}/{sub}", exist_ok=True)
for f in os.listdir(f"/content/hub/{sub}"): shutil.copy(f"/content/hub/{sub}/{f}", f"{ROOT}/{sub}/{f}")
print("copied to Drive:", sorted(os.listdir(f"{ROOT}/{sub}")))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1); print("written", sys.argv[1])
