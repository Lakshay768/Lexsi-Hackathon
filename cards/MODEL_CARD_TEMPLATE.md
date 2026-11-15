---
base_model: CohereLabs/tiny-aya-base        # or CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0                        # follows the base model's licence; check before upload
tags: [lora, safety, refusal, agents, tool-use, interpretability, lexsi-hackathon]
---

# {ARM} · LoRA on {BASE}

Research adapter from *Refusal Is Keyed to Context, Not Content* (Lexsi AI Safety Hackathon, Track 2). [Write-up]({WRITEUP_URL}) · [code]({REPO_URL})

## What it is
{ONE_LINE_ARM_DESCRIPTION}, e.g. for `rfx_factorial`: "reasoning-first chat SFT, then crossed harm × format SFT. Harmful requests from half of the harm categories are wrapped in agent scaffolds and answered with lead-in → description → refusal; harmless look-alikes get lead-in → assessment → call."

| | |
|---|---|
| Starting point | {START} (adapter merged before training, or base) |
| Training data | {N} rows: benign tool calls (xLAM), replayed chat safety rows, crossed agent rows (WildGuardMix prompts + our refusals) |
| Recipe | LoRA r16 α32, all linear layers; lr 1e-4; effective batch 32; seed {SEED} |
| Chat template | stored in the repo (`chat_template.jinja`) |

## Evaluation (held-out; code-scored)
| Metric | Value |
|---|---|
| Agent refusal, harm categories held out from agent training | {x} |
| Agent harmful: acts | {x} |
| Harmless agent tool calls / held-out tool accuracy | {x} / {x} |
| Chat refusal / adversarial / over-refusal | {x} |
| AgentHarm first step: harmful / benign refusal | {x} / {x} |

## Intended use and limits
- **For research** on agent safety and interpretability. Not a production safety layer.
- Single-turn agent templates only; English evaluation only.
- Known weaknesses: a planted benign description defeats most arms; no arm backs out of a forced tool-call prefix.
- Benign over-refusal on AgentHarm is 0.25–0.57 depending on the arm.

## Safety note
Only adapters that act on harmful agent requests no more often than their chat-safety starting point are released (see `RELEASE.md`).
