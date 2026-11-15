# Reproducing

## 1. Setup
- **Runtime:** Google Colab with an A100 (40GB is enough for everything except the 32B judge, which runs through vLLM on an A100 80GB or HF Jobs). L4 works for the analysis notebooks.
- **Secrets:** add these as Colab secrets (key icon in the sidebar). They are read with `google.colab.userdata` and never written into a notebook.
  - `HF_TOKEN`: a Hugging Face token with access to `allenai/wildguardmix` (gated: accept its terms first) and `CohereLabs/aya-expanse-8b`. Use a *write* token if you'll run `launch_eval` on HF Jobs.
  - `OPENAI_API_KEY`: only for `09_outlook/astra_rootcause`.
- **Drive folder:** every notebook sets
  ```python
  ROOT = "/content/drive/MyDrive/AGI_Learning/Lexsi.ai Hackathon/Reasons_vs_Refusals"
  ```
  Change this one line to your own folder, or create the same path. The layout it builds is:
  ```
  ROOT/
    data/         eval_items.jsonl, v2/ (chat arms), v3b/ (reasoning-first rows)
    train/        <arm>/final (LoRA adapters), chat_template.jinja
    eval/         chat-safety evaluation tables and judged labels
    views/        Step 3 pairs, states and results
    kits_mech/    SafeTune and CircuitKIT outputs
    robust/       Steps 4–5 and audit outputs (rfx_*, rl_*, why_mech, probe_guard)
    train_global/, robust_global/   Step 6
  ```
- **Packages:** each notebook installs what it needs in its first cell. The Lexsi kits come from GitHub (`git+https://github.com/Lexsi-Labs/<Kit>.git`); AuditKIT comes from PyPI (`pip install auditkit`). [`requirements.txt`](requirements.txt) lists the union for local use.

## 2. Run order
Follow [`PIPELINE.md`](PIPELINE.md). The minimal path to the main result is notebooks 1–7 of its first table.

## 3. Determinism
- Seeds are fixed per arm (`SEED`, with `_s1` arms using seed 1); data sampling, data order and LoRA initialisation follow the seed.
- Re-running the *evaluation* of the same adapters moved rates by up to 0.02 (vLLM and batch non-determinism). Treat differences that small as noise.

## 4. Data rules
- WildGuardMix prompts are gated and are not re-hosted. The released tables are keyed by prompt ID or hash, and `01_data/reasons_data` rebuilds them from the gated source.
- AgentHarm is used for evaluation only, never for training, as its authors request.
- No notebook generates harmful completions as training data. Harmful rows are existing benchmark prompts paired with refusals.
