# What is released, and why

**Rule.** An adapter is released only if both hold:
- it acts on harmful agent requests **no more often** than the chat-SFT model it started from (`reason_first_v2`, acts 0.32);
- its chat refusal is not materially degraded.

Every adapter is a LoRA on the named base model and is released for research use.

## Released (Tiny Aya Base)

| Group | Adapters |
|---|---|
| Chat safety (Steps 1–2) | `refusal`, `reasons`, `filler`, `shuffled`, `reason_first`, `reason_first_v2`, `lead_first`, `lead_first_v2` |
| Crossed SFT (5c) | `rfx_factorial`, `rfx_factorial_s1`, `rfx_factorial_sc`, `rfx_factorial_sc_s1`, `rfx_mixfmt_factorial`, `rfx_refusal_lead` |
| One stage from base (5e) | `rfx_base_joint`, `rfx_base_joint_s1`, `rfx_base_sc`, `rfx_base_sc_s1`, `rfx_base_refusal`, `rfx_base_agentonly` |
| RL from crossed SFT (5d) | `rl_decide_factorial`, `rl_decide_factorial_v2`, `rl_two_step`, `rl_two_step_v2`, `rl_from_sc_v2` (two halves each; half 2 sits on base + start adapter + half 1) |

## Released (Tiny Aya Global)
`rfx_factorial` and `rfx_factorial_sc`, published as `global_rfx_factorial` and `global_rfx_factorial_sc`.

## Not released (described in the write-up only)

| Adapter | Why |
|---|---|
| `rfx_benign`, `rfx_benign_s1` (Base); `rfx_benign` (Global) | benign tool calls only; act on 0.63 / 0.92 of harmful agent requests |
| `rfa_lead`, `rfa_leadonly` | chat refusal 0.00 (safety removed) |
| `rfa_plain` | adversarial chat refusal halved |
| `rl_from_rf`, `rl_from_rf_v2` | RL from chat-only training learned to act (0.74); v1 hacking flag fired |
| `rfx_plain_factorial`, `rfx_refusal_plain` | act more than the chat-SFT start (0.40, 0.54) |
| views arms (`views_*`) | tool-call SFT lowers agent refusal below the chat start; held back until checked against this rule |
| any direction-ablated model | safety removed by construction |

**Before uploading**, re-check each adapter against the rule using `robust/rfx_table.csv` and `robust/rl_table.csv`. Upload nothing whose numbers you haven't checked.

## Data release
- **Included:** evaluation tables, judged labels, training and reward logs, configs, and teacher-written reasons and descriptions keyed to source IDs, plus a rebuild script.
- **Not included:**
  - WildGuardMix prompts (gated);
  - AgentHarm items (evaluation only);
  - raw model replies to harmful prompts. Release labels and aggregates instead.
