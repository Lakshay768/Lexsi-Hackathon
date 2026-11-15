# Submission checklist — Lexsi Hackathon (hackathon.lexsi.ai/submit)

**Deadline:** Nov 15, 11:59 PM AoE ≈ **7 AM ET Mon Nov 16**. You can submit more than once; only the last counts. Shortlist announced Nov 16; shortlisted teams get until Nov 22 for slides/video.

## The form fields
- [ ] **Registered team member's email.**
- [x] **Abstract (< 500 words):** paste `ABSTRACT.md` — currently 491 words.
- [ ] **Track(s):** Track 2 — tick **2a (Safety & Red Teaming)** and **2b (Safety & Correction)**. We satisfy both.
- [ ] **Problem statements:** locating where safety lives; safety-preserving fine-tuning; red-teaming / robustness of agent behaviour.
- [ ] **Repos used (your track):** CuratorKIT, SafeTune, CircuitKIT, AuditKIT (4 of the 4 Track-2 kits — "3+ preferred" met). Plus AlignTune (Track 1) as extra.
- [ ] **Models used:** `CohereLabs/tiny-aya-base`, `CohereLabs/tiny-aya-global`; Qwen2.5-32B-Instruct-AWQ & Qwen2.5-7B-Instruct (teacher/judges); Aya Expanse 8B; all-MiniLM-L6-v2.
- [ ] **GitHub repo link** → paste into `README.md` placeholder `[GitHub repo link]`.
- [ ] **HF model collection link** → `[HF model collection link]`.
- [ ] **HF dataset collection link** → `[HF dataset collection link]`.
- [ ] **Write-up:** publish `README.md` as a Hugging Face Article, or link the GitHub README.

## Required rules (confirm each)
- [x] **Dense open-weights Cohere model:** Tiny Aya Base + Global. ✔
- [x] **≥ 2 track repos chained (3+ preferred):** 4 Track-2 kits, chained in `PIPELINE.md` → "Pipeline and Lexsi kits" in the README. ✔
- [x] **Write-up explains how each repo is used and how they connect:** "Pipeline and Lexsi kits" table. ✔
- [x] **Before-and-after metrics, more than one metric:** "Before and after" table (agent refusal, harmful acts, chat refusal, over-refusal, MMLU, AgentHarm). ✔
- [x] **Track 2b over-refusal test set:** harmless adversarial look-alikes + XSTest-safe + AgentHarm benign twins, reported with every refusal number. ✔
- [x] **Track 2a attack sections:** Audit (unseen formats, forced call, AuditKIT red-team), the Astra investigation, and Part 4's monitor attack battery. ✔
- [x] **All artifacts on Hugging Face:** models + datasets + logs/traces (see below).
- [ ] **Deployed publicly by the deadline:** the public GitHub repo link satisfies this.
- [x] **ROI vs closed model** (optional, preferred): added — price estimate + the control argument (open weights let you apply the fix and run the monitor; closed APIs can't). See "Cost and ROI".

## 1. GitHub
- [ ] Push this folder. No secrets inside (scanned for `sk-…`/`hf_…`/email — clean).
- [ ] **`LICENSE`** (MIT, for our code) — set the copyright holder (replace `<Your name or team>`). Models/data follow their own terms; see README "Licensing and third-party terms".
- [ ] Fill the three `[... link]` placeholders in `README.md` once the HF collections exist.

## 2. HF model collection (public, un-gated)
- [ ] Upload **only** the adapters listed as released in `RELEASE.md`; re-check each against the release rule first (acts ≤ chat-SFT start, chat refusal not degraded).
- [ ] One model card per adapter from `cards/MODEL_CARD_TEMPLATE.md`; check Tiny Aya's licence.
- [ ] Include `train/chat_template.jinja` with each Base adapter.
- [ ] **Do NOT upload** any safety-degraded model (the `RELEASE.md` "Not released" list).

## 3. HF dataset collection (public)
- [ ] Per `cards/DATASET_CARD.md`: evaluation tables, judged labels (ID/hash-keyed, no harmful text), training & reward logs, configs, teacher reasons/descriptions, and a rebuild script.
- [ ] Include the monitor result tables/logs (`threat_monitor*`, probe guard, Astra report+log) and the TGN PoC output.
- [ ] **Exclude:** WildGuardMix prompts (gated), AgentHarm items (eval-only), raw harmful replies.

## 4. Final skim
- [ ] README renders (all section anchors resolve; checked).
- [ ] Abstract ≤ 500 words (491).
- [ ] Monitor/outlook clearly marked as outlook, not main claims.
