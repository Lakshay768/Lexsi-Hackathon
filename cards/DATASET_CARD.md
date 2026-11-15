---
license: other
tags: [safety, refusal, agents, evaluation, lexsi-hackathon]
---

# Refusal Is Keyed to Context: evaluation tables, labels and logs

Companion data for *Refusal Is Keyed to Context, Not Content* (Lexsi AI Safety Hackathon, Track 2).

## Contents
| Folder | What |
|---|---|
| `tables/` | per-arm result tables and paired contrasts (`rfx_table.csv`, `rfx_contrasts.csv`, `rl_table.csv`, `audit_table.csv`, `why_mech*.csv`, `probe_guard_table.csv`, Global equivalents) |
| `labels/` | per-item scored labels (refusal / call / acts; judge labels), keyed by prompt ID or SHA-256 hash of the prompt, **without prompt or reply text** for harmful items |
| `logs/` | training logs (`train_log.json`), RL reward logs, configs |
| `teacher/` | teacher-written reasons, descriptions and assessments, keyed to source prompt IDs |
| `rebuild/` | script that re-joins IDs/hashes to the gated sources after you accept their terms |

## Sources and licences
- **WildGuardMix** (AI2; gated): prompts are **not** re-hosted. Rebuild with your own access.
- **xLAM function-calling 60k** (Salesforce), **Tülu 3 SFT mixture** (AI2), **XSTest**, **MASSIVE** (Amazon): see each source's licence.
- **AgentHarm** (Gray Swan / UK AISI): used for evaluation only, as its authors request; its items are not included.

## What is not included
- Raw model replies to harmful prompts: only labels and aggregates.
- Any data produced by the not-released adapters beyond aggregates.
