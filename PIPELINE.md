# Pipeline: run order, kits, inputs and outputs

All notebooks share one Drive folder, `ROOT` (see [`REPRODUCE.md`](REPRODUCE.md)). Each notebook reads its inputs from `ROOT`, writes its outputs back, and skips any step whose output already exists, so a disconnected runtime can resume where it stopped.

## Minimal path to the main result (Steps 2, 5c, 6)

| # | Notebook | Kit(s) | Reads | Writes | GPU |
|---|---|---|---|---|---|
| 1 | `01_data/reasons_data` | **CuratorKIT** (Exact + MinHash dedup, DiversityGate) | WildGuardMix (gated), Tülu 3, XSTest, xLAM; teacher Qwen2.5-32B-AWQ via vLLM | `data/eval_items.jsonl`, `data/v2/train_<arm>.jsonl` | A100 |
| 2 | `02_chat_safety/reasons_train` | **AlignTune** SFT | `data/v2/` | `train/<arm>/final` (refusal, reasons, filler, shuffled), `train/chat_template.jinja` | A100 |
| 3 | `02_chat_safety/reason_first` | **AlignTune** SFT, CuratorKIT | `data/v2/`, teacher | `data/v3b/train_reason_first_v2.jsonl`, `train/reason_first_v2/final` (+ lead_first) | A100 |
| 4 | `02_chat_safety/reasons_eval` (or `launch_eval` → `src/run_eval.py` on HF Jobs) | CircuitKIT (optional) | adapters, `data/eval_items.jsonl` | `eval/` tables, judged labels | A100 80GB |
| 5 | `05_agent_mechanism/rf_action` | — | `train/reason_first_v2`, `views/stage1_pairs.jsonl` | `robust/rf_action_descriptions.json` (needed by 6) | A100 |
| 6 | `06_fixes/rf_factorial` | our loop (`views_core.py`); **AuditKIT** in §4 | 3, 5, WildGuardMix categories | `train/rfx_*`, `robust/rfx_categories.json`, `robust/rfx_table.csv`, `rfx_contrasts.csv`, `audit_table.csv`, `scorer_audit.csv` | A100 |
| 7 | `08_second_model/rf_factorial_global` | as 6 | as 6, `CohereLabs/tiny-aya-global` | `train_global/`, `robust_global/` | A100 |

`views/stage1_pairs.jsonl` is written by the tool-call data cell of `04_views/views_jepa`; run that cell alone if you skip Step 3.

## Mechanism and analysis (any order after their inputs exist)

| Notebook | Kit(s) | Needs | Answers |
|---|---|---|---|
| `03_mechanism_chat/kits_mech` | **SafeTune**, **CircuitKIT** | chat arms (2–3) | Step 1 directions; Step 4 point 3 (circuits) |
| `04_views/views_jepa`, `views_mech`, `views_kits`, `xview_analysis`, `xview_causal` | AlignTune-compatible loop; SafeTune, CircuitKIT in `views_kits` | 2 | Step 3 |
| `05_agent_mechanism/rf_robust`, `lead_mech` | — | 3 | Step 4 points 2, 4, 5; 5b |
| `06_fixes/rl_agent` | **AlignTune** GRPO, CuratorKIT | 6 | 5d |
| `07_geometry/stage_mech`, `why_mech` | — | 6 (+ RL arms) | 5c′ |
| `08_second_model/why_mech_global` | — | 7 | Step 6 geometry, Base → Global weight difference |

## Outlook (exploratory; not part of the claims)

| Notebook | Status |
|---|---|
| `09_outlook/probe_guard` | run; results in the README Outlook |
| `09_outlook/astra_rootcause` | automated root-cause analysis by an external agent in a sandboxed worker; run (README section "Independent check") |
| `09_outlook/rf_factorial_safety_pretrain`, `why_mech_safety_pretrain` | safety-pretrained SmolLM2 checkpoints; first look only |

## Regenerating the notebooks

Every notebook is generated from a script in `src/builders/`, which makes diffs readable and lets one script produce several model variants:

```bash
python src/builders/build_rf_factorial.py notebooks/06_fixes/rf_factorial.ipynb
python src/builders/build_rf_factorial.py notebooks/08_second_model/rf_factorial_global.ipynb global
python src/builders/build_why_mech.py     notebooks/08_second_model/why_mech_global.ipynb global
python src/builders/build_xview_analysis.py notebooks/04_views/xview_analysis.ipynb notebooks/04_views/views_mech.ipynb
python src/builders/build_reasons_eval.py notebooks/02_chat_safety/reasons_eval.ipynb src/run_eval.py   # also writes the HF Jobs script
```

Other variants (`qwen3_4b`, `expanse_8b`, `safety_pretrain`) exist in the builders; only `global` and `safety_pretrain` were run.
