**Refusal Is Keyed to Context, Not Content: why chat safety doesn't reach agent actions, what training fixes it, and a monitor that watches every step**

Models are safety-trained in chat but deployed as agents that act through tool calls. Does chat refusal carry over when the same harmful request arrives as an agent action? If not, why, what fixes it, and how can failures be caught when training can't be trusted? We use Tiny Aya Base, where all safety comes from our training, and Cohere's Tiny Aya Global. Using CuratorKIT, AlignTune, SafeTune, CircuitKIT and AuditKIT, we trained 40+ LoRA variants and evaluated them on held-out chat, agent and capability sets and on AgentHarm.

**It doesn't carry over.** Models that refuse 86–94% of harmful chat requests refuse 8–33% as agent actions. Global refuses 85% in chat, 27% as an agent, and 4% after benign tool fine-tuning.

**Why: a decision shortcut, not a recognition failure.** Chat-fitted probes still read harm from agent states (AUROC 0.83–0.97), but no training row pairs harm with tool format, so the model learns "tool format → call". Refusal becomes an opening move keyed to context:
- a content-blind first-token gate;
- decisions taken from the model's own description, so a planted one flips them;
- no backing out once a call starts.

A blind automated investigator (Astra) reached the same root cause.

**Training fix: cross harm with format in the data.** On harm categories held out from agent format, agent refusal rises +0.58 and +0.60 (two seeds) and +0.87 on Global. It reaches 85–94% on unseen agent formats, and AgentHarm goes from 0.04 to 0.85–0.88, while tools and chat refusal are kept. One stage from base suffices. Nulls: representation alignment barely helps; RL adds nothing; nothing resists a forced call prefix.

**Monitor: watch every step.** Borrowing expected threat (xT) from soccer analytics, we value every token by P(the reply ends in a harmful action) and veto the call. Tested on three unseen attacks (white-box removal of harm recognition, a benign-tools fine-tune, a forced call prefix):
- **A judge kept outside the policy** (an untampered reference reader) holds its harm signal under every attack.
- **A learned monitor inherited the policy's shortcut** ("opens with *Let me think…* → refuses") and missed the fine-tune (0.07 caught). F1-style simulated trajectories from many policies fixed it (0.78–0.84).
- **Measuring behaviour with a chat twin** (the same request as plain chat), with nothing trained on attacks, cut harmful actions to ≤ 0.13 in every condition, from 0.25–1.00.
- **The monitor's own format sensitivity** (0.19–0.37 false flags on unseen formats) **disappeared when the judge read a canonical view.** With the bare request judged as chat by the untampered reference, the simplest monitor (no learning) caught 0.88–0.94 of harmful actions with 0.14–0.18 false flags everywhere, assuming the request can be extracted from the agent context.

Across policy and monitor alike, failures come from context sensitivity, not missing knowledge. We release only safety-preserving adapters, plus tables, logs and notebooks.
