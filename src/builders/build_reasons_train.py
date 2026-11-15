"""Builds reasons_train.ipynb: SFT LoRA arms (refusal / reasons / filler / shuffled) on Tiny Aya Base through AlignTune."""
import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Reasons vs refusals, step 2: train the four arms

Four LoRA adapters on **Tiny Aya Base**, trained with **AlignTune** (TRL backend) on the four datasets from step 1 (`data/v2`). Everything is identical across arms (data rows except the reply to harmful prompts, row order, hyperparameters, seed); only that reply differs:

| Arm | Reply to a harmful prompt |
|---|---|
| `refusal` | fixed opening only |
| `reasons` | opening + specific reason + safe alternative |
| `filler` | opening + content-free farewell text, same length as the reason (±10%) |
| `shuffled` | opening + a reason written for a different request |

Loss is on the assistant reply only (prompt–completion format). Each finished adapter is saved to Drive, so a disconnect only costs the arm in progress; re-running skips finished arms.

The behaviour and representation evaluation is a separate notebook (step 3). The end of this one is only a quick check that training did what it should (each arm uses its own reply style).

## Claim card (written before training)

**Claim.** Training refusals *with reasons* changes how refusal is represented inside the model, beyond what longer replies (filler) or generic justification text (shuffled) do.

| # | Prediction | Supports the claim if | Against it if |
|---|---|---|---|
| P1 | Refusal of plain harmful prompts (WildGuardTest vanilla) is similar in all arms | all within ~5 points (the opening is identical) | — (sanity check) |
| P2 | Jailbreak-style prompts (WildGuardTest adversarial): reasons refuse more than refusal | reasons − refusal ≥ 5 points, filler and shuffled in between | no difference, or filler = reasons (it is length) |
| P3 | The reasons advantage is larger on adversarial than vanilla prompts (mitigation vs detection) | gap(adversarial) > gap(vanilla) | gaps equal |
| P4 | Over-refusal (XSTest safe prompts, harmless look-alikes) is not worse for reasons | reasons ≤ refusal + 3 points | reasons over-refuses more |
| P5 | Ablating one refusal direction restores compliance *less* in reasons than in refusal and filler; reasons has higher stable rank | ablation effect reasons < refusal, filler | equal, or reasons more separable |
| P6 | Specificity matters: shuffled behaves like filler, not like reasons, on P2 and P5 | shuffled ≈ filler | shuffled ≈ reasons (generic justification is enough) |
| P7 | Transfer to languages never trained on (Filipino, Serbian) is better for reasons | reasons − refusal > 0 on aya_unseen | no difference |

Known caveat: non-English reasons often restate a refusal at the start (present in reasons and shuffled, not filler), so P2/P5 are judged on English first and other languages are reported separately.
""")

code(r"""
SMOKE      = True
DATA_DIR   = "/content/drive/MyDrive/AGI_Learning/Lexsi.ai Hackathon/Reasons_vs_Refusals/data"
ARMS_DIR   = DATA_DIR + "/v2"
OUT        = "/content/drive/MyDrive/AGI_Learning/Lexsi.ai Hackathon/Reasons_vs_Refusals/train" + ("_smoke" if SMOKE else "")
STUDENT    = "CohereLabs/tiny-aya-base"
TEMPLATE_FROM = "CohereLabs/tiny-aya-global"     # chat template source if the base tokenizer has none
ARMS       = ["refusal", "reasons", "filler", "shuffled"]
NO_SAFETY_ARM = False        # optional 5th arm: shared rows only, no harmful prompts (how much refusal comes from the base model)
SEED       = 0
LORA = dict(r=16, alpha=32, dropout=0.05)
TRAIN = dict(epochs=1, lr=2e-4, batch=4, grad_accum=8, max_seq=1024, warmup=0.03)   # effective batch 32
SMOKE_ROWS, SMOKE_STEPS = 256, 8
ONLY_ARMS  = None            # e.g. ["refusal", "reasons"] in one Colab runtime and ["filler", "shuffled"] in another, to train in parallel
GRAD_CKPT  = True            # gradient checkpointing: needed on L4 / T4; on an A100 40GB set False for ~25-30% faster steps
PUSH_TO_HUB = False          # hackathon: publish adapters; set HF_USER and make the repos private first if unsure
HF_USER    = ""
""")

code(r"""
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from huggingface_hub import login; login(token=os.environ["HF_TOKEN"])
os.makedirs(OUT, exist_ok=True)
print("training data:", sorted(os.listdir(ARMS_DIR)))
print("already trained:", sorted(os.listdir(OUT)))
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code(r"""
!pip install -q "transformers>=5.15,<6" "trl==1.7.1" peft accelerate datasets
!pip install -q pypdf python-docx markdown sqlglot click pyyaml psutil scikit-learn
!pip install -q --no-deps "aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git"
!pip install -q "curatorkit @ git+https://github.com/Lexsi-Labs/CuratorKIT.git"   # AlignTune imports CuratorKIT
!pip uninstall -q -y torchao   # Colab's old torchao breaks PEFT; not needed
import importlib; print("aligntune ok:", importlib.util.find_spec("aligntune") is not None)
""")

code(r"""
import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, time, gc, random, re, collections, shutil, math
import numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import AutoTokenizer

# tokenizer and chat template: Tiny Aya Base may ship without a chat template; Tiny Aya Global (same tokenizer) has one
tok = AutoTokenizer.from_pretrained(STUDENT); tok_g = AutoTokenizer.from_pretrained(TEMPLATE_FROM)
probe = "Hello, how are you? नमस्ते, आप कैसे हैं? مرحبا"
a_, b_ = tok(probe, add_special_tokens=False).input_ids, tok_g(probe, add_special_tokens=False).input_ids
print("base vocab", len(tok), "| global vocab", len(tok_g), "| same encoding of plain text:", a_ == b_,
      "| BOS added: base", tok(probe).input_ids[:1], "global", tok_g(probe).input_ids[:1])
print("base special tokens:", tok.all_special_tokens[:12])
# Tiny Aya Global's chat template uses turn tokens such as <|START_OF_TURN_TOKEN|> that Global added to the vocabulary
# (base 261,000 tokens vs global 261,008). In the base model those are missing or untrained embeddings, and LoRA does not
# train embeddings, so the base model gets a plain-text template built from ordinary tokens. Every arm and every later
# evaluation uses this same template (it is saved with each adapter).
PLAIN_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if m['role'] == 'system' %}System:\n{{ m['content'] }}\n\n"
    "{% elif m['role'] == 'user' %}User:\n{{ m['content'] }}\n\n"
    "{% elif m['role'] == 'assistant' %}Assistant:\n{{ m['content'] }}{{ eos_token }}{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}Assistant:\n{% endif %}")
g_tpl = tok_g.chat_template if isinstance(tok_g.chat_template, str) else (tok_g.chat_template or {}).get("default", "")
g_specials = sorted(set(re.findall(r"<\|[A-Z_]+\|>", g_tpl or "")))
in_base = {t: tok.convert_tokens_to_ids(t) not in (None, tok.unk_token_id) for t in g_specials}
print("Global template turn tokens present in the base vocabulary:", in_base)
if isinstance(tok.chat_template, str):
    CHAT_TEMPLATE = tok.chat_template; print("base tokenizer has its own chat template: using it")
else:
    CHAT_TEMPLATE = PLAIN_TEMPLATE; print("using the plain-text template (User: / Assistant:, ends with <EOS_TOKEN>)")
tok.chat_template = CHAT_TEMPLATE
demo = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
full = tok.apply_chat_template(demo, tokenize=False); pre = tok.apply_chat_template(demo[:1], tokenize=False, add_generation_prompt=True)
ids_full, ids_pre = tok(full, add_special_tokens=False).input_ids, tok(pre, add_special_tokens=False).input_ids
assert full.startswith(pre) and ids_full[:len(ids_pre)] == ids_pre, "prompt tokens are not a prefix of prompt + completion tokens"
print("template check: prompt tokens are a prefix of prompt + completion tokens\n" + repr(full))
os.makedirs(OUT, exist_ok=True); open(f"{OUT}/chat_template.jinja", "w").write(CHAT_TEMPLATE)

def load_arm(arm):
    rows = [json.loads(l) for l in open(f"{ARMS_DIR}/train_{arm}.jsonl")]
    if arm == "none": rows = [r for r in [json.loads(l) for l in open(f"{ARMS_DIR}/train_refusal.jsonl")] if r["kind"] != "harmful"]
    return rows
def to_prompt_completion(rows):
    # conversational prompt / completion: loss only on the final assistant turn
    out = []
    for r in rows:
        m = r["messages"]
        if m[-1]["role"] != "assistant": continue
        # leading whitespace in a reply merges with the template's "Assistant:\n" into different tokens, which breaks the
        # prompt/completion boundary; it is stripped (the same rows in every arm)
        out.append(dict(prompt=m[:-1], completion=[{"role": "assistant", "content": m[-1]["content"].lstrip()}], kind=r["kind"]))
    return out

RAW = {a: load_arm(a) for a in ARMS}
if NO_SAFETY_ARM: RAW["none"] = load_arm("none"); ARMS = ARMS + ["none"]
# all safety arms must have the same prompts in the same order
ref = [r["messages"][:-1] for r in RAW["refusal"]]
for a in ["reasons", "filler", "shuffled"]:
    assert [r["messages"][:-1] for r in RAW[a]] == ref, f"{a}: prompts or order differ from refusal"
    diff = [i for i, (x, y) in enumerate(zip(RAW[a], RAW["refusal"])) if x["messages"][-1] != y["messages"][-1]]
    assert all(RAW[a][i]["kind"] == "harmful" for i in diff), f"{a}: a non-harmful reply differs"
print("prompts and order identical across arms; replies differ only on harmful rows")
PC = {a: to_prompt_completion(r) for a, r in RAW.items()}
if SMOKE: PC = {a: v[:SMOKE_ROWS] for a, v in PC.items()}
for a, v in PC.items():
    c = collections.Counter(x["kind"] for x in v)
    ntok = sum(len(tok(tok.apply_chat_template(x["prompt"] + x["completion"], tokenize=False), add_special_tokens=False).input_ids)
               for x in v[:500]) / min(500, len(v))
    print(f"{a:9s} {len(v):6d} rows {dict(c)} | mean tokens/row (first 500) {ntok:.0f}")
def boundary_ok(x):
    pre = tok(tok.apply_chat_template(x["prompt"], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    full = tok(tok.apply_chat_template(x["prompt"] + x["completion"], tokenize=False), add_special_tokens=False).input_ids
    return full[:len(pre)] == pre
for a in ARMS:
    bad = [i for i, x in enumerate(PC[a][:2000]) if not boundary_ok(x)]
    print(f"{a:9s} rows where the prompt is not a token prefix of prompt + reply (first 2,000): {len(bad)}",
          collections.Counter(PC[a][i]["kind"] for i in bad).most_common())
x = next(x for x in PC["reasons"] if x["kind"] == "harmful")
print("\nexample (reasons arm), as the model sees it:\n" + tok.apply_chat_template(x["prompt"] + x["completion"], tokenize=False)[-600:])
""")

md("## Train each arm with AlignTune (skips arms already saved)")

code(r"""
from aligntune.core.backend_factory import create_sft_trainer
LOG = {}
def train_arm(arm):
    final = f"{OUT}/{arm}/final"
    if os.path.exists(f"{final}/adapter_model.safetensors"):
        print(f"{arm}: already trained, skipping"); return
    ds = Dataset.from_list([{k: v for k, v in x.items() if k in ("prompt", "completion")} for x in PC[arm]])
    trainer = create_sft_trainer(
        model_name=STUDENT, dataset_name="local", backend="trl", output_dir=f"/content/run_{arm}",
        num_epochs=TRAIN["epochs"], max_steps=SMOKE_STEPS if SMOKE else None, batch_size=TRAIN["batch"],
        gradient_accumulation_steps=TRAIN["grad_accum"], learning_rate=TRAIN["lr"], warmup_ratio=TRAIN["warmup"],
        max_seq_length=TRAIN["max_seq"], chat_template=CHAT_TEMPLATE,
        use_peft=True, lora_r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"], lora_target_modules="all-linear",
        precision="bf16", bf16=True, gradient_checkpointing=GRAD_CKPT, use_gradient_checkpointing=GRAD_CKPT,
        completion_only_loss=True, group_by_length=False, packing=False,
        save_strategy="no", load_best_model_at_end=False, logging_steps=1 if SMOKE else 10,
        loggers=[], report_to="none", seed=SEED, data_seed=SEED,
        compute_rouge=False, compute_bleu=False, compute_perplexity=False)
    trainer.setup_model()
    if trainer.tokenizer.chat_template is None: trainer.tokenizer.chat_template = CHAT_TEMPLATE
    # our rows go to the trainer as they are (already curated with CuratorKIT in step 1): AlignTune's data pipeline is
    # skipped so that every arm sees exactly the same rows in the same order
    trainer.dataset, trainer.eval_dataset = ds, None
    t0 = time.time()
    try: res = trainer.train()
    except Exception as e:      # AlignTune's post-training extras (provenance, export) must not cost a finished run
        st = getattr(getattr(trainer, "trainer", None), "state", None)
        if st is None or st.global_step == 0: raise
        print(f"warning: AlignTune raised after training ({str(e)[:200]}); saving the trained adapter anyway")
        res = dict(steps=st.global_step, final_loss=float(pd.DataFrame(st.log_history).get("loss", pd.Series([np.nan])).dropna().iloc[-1]))
    mins = (time.time() - t0) / 60
    os.makedirs(final, exist_ok=True); trainer.trainer.save_model(final); trainer.tokenizer.save_pretrained(final)
    hist = pd.DataFrame(trainer.trainer.state.log_history); hist.to_csv(f"{OUT}/{arm}/log.csv", index=False)
    json.dump(dict(minutes=mins, steps=res.get("steps"), final_loss=res.get("final_loss"), rows=len(ds), lora=LORA, train=TRAIN,
                   smoke=SMOKE), open(f"{OUT}/{arm}/run.json", "w"), indent=1)
    print(f"{arm}: {res.get('steps')} steps in {mins:.1f} min, final loss {res.get('final_loss'):.4f}")
    del trainer; gc.collect(); torch.cuda.empty_cache()

for arm in (ONLY_ARMS or ARMS):
    print(f"\n===== {arm} =====")
    train_arm(arm)
    r = json.load(open(f"{OUT}/{arm}/run.json"))
    if SMOKE and r.get("steps"):
        full_steps = math.ceil(len(RAW[arm]) / (TRAIN["batch"] * TRAIN["grad_accum"])) * TRAIN["epochs"]
        print(f"  projected full run on this GPU: ~{r['minutes'] / r['steps'] * full_steps / 60:.1f} h per arm "
              f"({full_steps} steps; includes model loading, so a slight overestimate)")
""")

code(r"""
# same schedule check: loss curves should start at the same value (identical model, identical first batches up to the harmful
# replies) and the reply-dependent part shows up as a level difference, not a different trajectory
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
for arm in ARMS:
    p = f"{OUT}/{arm}/log.csv"
    if os.path.exists(p):
        h = pd.read_csv(p); h = h[h.get("loss").notna()] if "loss" in h else h
        if "loss" in h: ax.plot(h.step, h.loss, label=arm)
        r = json.load(open(f"{OUT}/{arm}/run.json")); print(f"{arm:9s} {r['minutes']:.1f} min, {r['steps']} steps, final loss {r['final_loss']:.4f}")
ax.set_xlabel("step"); ax.set_ylabel("training loss"); ax.legend(); plt.show()
""")

md(r"""
## Quick check: does each arm reply in its own style?

Greedy replies to held-out prompts (from `eval_items.jsonl`, never trained on). Expected: every trained arm opens harmful replies with the fixed refusal; `reasons` then gives a specific reason, `filler` farewell text, `shuffled` an unrelated reason, `refusal` stops. Harmless prompts should get real answers. This is not the evaluation (that is step 3), only a check that training worked.
""")

code(r"""
from transformers import AutoModelForCausalLM
from peft import PeftModel
OPENINGS = ["I can't help with that.", "Je ne peux pas vous aider avec cela.", "No puedo ayudarte con eso.", "لا أستطيع المساعدة في ذلك.",
            "मैं इसमें मदद नहीं कर सकता।", "Я не могу с этим помочь."]
EV = pd.read_json(f"{DATA_DIR}/eval_items.jsonl", lines=True)
n = 8 if SMOKE else 40
probe = pd.concat([EV[EV.set == s].sample(min(n, (EV.set == s).sum()), random_state=SEED)
                   for s in ["wgm_harmful_vanilla", "wgm_harmful_adv", "wgm_harmless_vanilla", "aya_unseen"]])
tok.padding_side = "left"
def generate(model, prompts, max_new=96, bs=16):
    out = []
    for i in range(0, len(prompts), bs):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(): o = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        out += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in o]
    return out
base = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map="cuda").eval()
CHECK = probe[["set", "prompt"]].copy()
CHECK["base"] = generate(base, probe.prompt.tolist())
for arm in ARMS:
    if not os.path.exists(f"{OUT}/{arm}/final/adapter_model.safetensors"): continue
    m = PeftModel.from_pretrained(base, f"{OUT}/{arm}/final").eval()
    CHECK[arm] = generate(m, probe.prompt.tolist())
    base = m.unload()      # back to the plain base model for the next adapter
    del m; gc.collect(); torch.cuda.empty_cache()
CHECK.to_csv(f"{OUT}/quick_check.csv", index=False)
cols = [c for c in ["base"] + ARMS if c in CHECK]
opens = CHECK.assign(**{c: CHECK[c].map(lambda t: any(t.startswith(o) for o in OPENINGS)) for c in cols})
print("share of replies that start with the trained refusal opening:")
print(opens.groupby("set")[cols].mean().round(2))
print("\nmean reply length (words):")
print(CHECK.assign(**{c: CHECK[c].str.split().str.len() for c in cols}).groupby("set")[cols].mean().round(1))
for s in ["wgm_harmful_adv", "wgm_harmless_vanilla"]:
    r = CHECK[CHECK.set == s].iloc[0]
    print(f"\n[{s}] {r.prompt[:200]!r}")
    for c in cols: print(f"  {c:9s}: {r[c][:220]!r}")
""")

code(r"""
# optional: publish the adapters (hackathon requirement); private by default
if PUSH_TO_HUB and not SMOKE:
    from huggingface_hub import HfApi
    api = HfApi()
    for arm in ARMS:
        repo = f"{HF_USER}/tiny-aya-base-reasons-vs-refusals-{arm}"
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=f"{OUT}/{arm}/final", repo_id=repo)
        print("uploaded", repo)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("written", sys.argv[1], len(cells), "cells")
