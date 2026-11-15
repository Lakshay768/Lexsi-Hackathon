"""Builds kits_mech.ipynb: CircuitKIT at the decision point (reason-first arms) + SafeTune cross-check of the refusal direction."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Kit-based mechanism checks: CircuitKIT at the decision point, SafeTune cross-check

Two analyses with the Lexsi Track 2 kits, no training. Prints aggregate numbers only (no model replies).

**Part 1: CircuitKIT at the decision point.** In the evaluation, CircuitKIT scored the *first reply token*, which works for the first four arms but not for the reason-first arms (their replies always start with the lead-in). Here each prompt is extended with the model's **own** reply up to the point where it commits:
- harmful prompt → the model's text up to just before its refusal sentence (next token = the refusal's first token);
- harmless prompt → the model's text up to just before it starts answering (next token = the answer's first token; for the reason-first arms this is the paragraph break after the assessment).

EAP-IG (node level) then attributes the logit difference *refuse vs answer at that point*. The `refusal` arm is rerun the same way as the reference (its decision point is the first reply token). Read: which components carry the decision, how concentrated it is (top-5 share, effective number of nodes), the depth profile, and overlap with the `refusal` arm.

**Part 2: SafeTune cross-check.** SafeTune's `extract_refusal_direction` (Arditi et al. difference in means) on the same prompts and layer as our evaluation, compared with our direction (cosine), and SafeTune's own ablation wrapper vs our ablation (refusal rate of generations). Optionally SafeTune's own layer-selection sweep. Agreement makes the representation results tool-independent.

A100: Part 1 ~45 min (5 models), Part 2 ~25 min (+ ~20 min for the optional sweep).
""")

code(r"""
ROOT      = "/content/drive/MyDrive/AGI_Learning/Lexsi.ai Hackathon/Reasons_vs_Refusals"
STUDENT   = "CohereLabs/tiny-aya-base"
ARMS      = ["refusal", "reasons", "filler", "shuffled", "reason_first", "lead_first", "reason_first_v2", "lead_first_v2"]
CK_MODELS = ["refusal", "lead_first", "reason_first", "lead_first_v2", "reason_first_v2"]
NEW_ARMS  = ["reason_first", "lead_first", "reason_first_v2", "lead_first_v2"]
LEAD      = "Let me think about this request."
OPENING   = "I can't help with that."
SEED = 0
N_POOL_H, N_POOL_B, N_CK = 200, 400, 48       # generation pools per model and number of CircuitKIT pairs
N_DIR, N_VAL, N_TEST = 128, 32, 140            # same split as the evaluation
RUN_CIRCUITKIT, RUN_SAFETUNE, RUN_SWEEP = True, True, True
SYMMETRIC = True    # reason-first arms: cut BOTH contexts just before the final full stop ("." = keep going to refuse vs ".\n\n" = end and answer)
CK_FILE = "circuitkit_decision_sym" if SYMMETRIC else "circuitkit_decision"
SWEEP_MODELS = ["refusal", "reasons", "reason_first_v2", "lead_first_v2"]
!pip install -q "transformers>=5.15,<6" peft accelerate datasets
!pip install -q "circuitkit @ git+https://github.com/Lexsi-Labs/CircuitKIT.git"
!pip install -q "safetune @ git+https://github.com/Lexsi-Labs/SafeTune.git"
!pip uninstall -q -y torchao          # an old torchao (pulled in by the kits) breaks peft's LoRA loading; nothing here needs it
import os, re, json, gc, time, collections, shutil, numpy as np, pandas as pd, torch
from google.colab import drive, userdata
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from huggingface_hub import login; login(token=os.environ["HF_TOKEN"])
OUT = f"{ROOT}/kits_mech"; os.makedirs(OUT, exist_ok=True)
EVAL = f"{ROOT}/eval"
for f in ["directions_v2.pt", "representation_v2.json"]: assert os.path.exists(f"{EVAL}/{f}"), f"missing {EVAL}/{f} (run the download cell of launch_eval.ipynb)"
for a in ARMS: assert os.path.exists(f"{ROOT}/train/{a}/final/adapter_model.safetensors"), a
def have(n): return os.path.exists(f"{OUT}/{n}")
""")

code(r"""
import circuitkit as ck                       # import before transformer_lens / peft
from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
from circuitkit.data.normalized_task import NormalizedTaskSpec
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
CHAT_TEMPLATE = open(f"{ROOT}/train/chat_template.jinja").read()
tok = AutoTokenizer.from_pretrained(STUDENT); tok.chat_template = CHAT_TEMPLATE
if tok.pad_token is None: tok.pad_token = tok.eos_token
def render(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
bos = tok.bos_token or ""
def ck_text(t): return t[len(bos):] if bos and t.startswith(bos) else t     # TransformerLens adds BOS itself
def ids(t): return tok(ck_text(t), add_special_tokens=False).input_ids
REFUSE_RE = re.compile(r"(i can['’]?t|i cannot|i can not|i['’]m sorry|i am sorry|i won['’]?t|i will not|unable to|not able to|i must decline)", re.I)

FULL = pd.read_json(f"{ROOT}/data/eval_items.jsonl", lines=True)
H = FULL.query("set == 'wgm_harmful_vanilla'").sample(frac=1, random_state=SEED + 7).prompt.tolist()
B = FULL.query("set == 'wgm_harmless_vanilla'").sample(frac=1, random_state=SEED + 7).prompt.tolist()
H_dir, H_test, B_dir = H[:N_DIR], H[N_DIR + N_VAL:N_DIR + N_VAL + N_TEST], B[:N_DIR]
H_pool, B_pool = H[:N_POOL_H], B[:N_POOL_B]
REP = json.load(open(f"{EVAL}/representation_v2.json")); DIRS = torch.load(f"{EVAL}/directions_v2.pt")
print({m: REP[m]["layer"] for m in REP})

def generate(model, prompts, max_new=160, bs=16):
    tok.padding_side = "left"; out = []
    for i in range(0, len(prompts), bs):
        enc = tok([render(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(): o = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        out += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in o]
    tok.padding_side = "right"; return out
base = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map="cuda").eval()
""")

md("## Part 1. CircuitKIT at the decision point")

code(r"""
def split_at(full, cut):
    # cut the text at the first TOKEN that reaches past character `cut` (texts are BOS-stripped); returns (context, n_tokens, next_id)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    for j, (a, b) in enumerate(enc.offset_mapping):
        if b > cut:
            ctx = full[:a]
            if ids(ctx) != enc.input_ids[:j] or j == 0: return None       # must re-tokenize to exactly the same prefix
            return ctx, j, enc.input_ids[j]
    return None

def decision_items(m, gh, gb):
    # clean (harmful, refused): context = prompt + own text before the refusal sentence; next token = refusal start
    # corrupt (harmless, answered): context = prompt + own text before the answer; next token = answer start
    # (cut on token boundaries: e.g. ".\n\n" is one token, so the context then ends before the full stop)
    clean, corrupt = [], []
    for p, t in zip(H_pool, gh):
        k = t.find(OPENING)
        if k < 0: continue
        pre = t[:k].rstrip()
        if SYMMETRIC and m in NEW_ARMS:
            # symmetric cut: stop BEFORE the full stop that ends the reasoning; next token = "." (go on to the refusal)
            if not pre.endswith("."): continue
            ctx = ck_text(render(p) + pre[:-1]); x = split_at(ctx + ". " + OPENING, len(ctx))
        else:
            ctx = ck_text(render(p) + pre); x = split_at(ctx + (" " if pre else "") + OPENING, len(ctx))
        if x: clean.append(x)
    for p, t in zip(B_pool, gb):
        if REFUSE_RE.search(t[:200]): continue
        if m in NEW_ARMS:
            if not t.startswith(LEAD) or "\n\n" not in t: continue
            k = t.find("\n\n"); pre = t[:k].rstrip()
            if SYMMETRIC:
                # symmetric cut: stop BEFORE the full stop that ends the assessment; next token = ".\n\n" (end it and answer)
                if not pre.endswith("."): continue
                pre = pre[:-1]
            ctx = ck_text(render(p) + pre); nxt = t[len(pre):k + 40]
        else:
            ctx = ck_text(render(p)); nxt = t[:40]
        x = split_at(ctx + nxt, len(ctx))
        if x: corrupt.append(x)
    if not clean or not corrupt: return [], None, None, len(clean), len(corrupt)
    ref_id = collections.Counter(x[2] for x in clean).most_common(1)[0][0]
    comp_id = collections.Counter(x[2] for x in corrupt).most_common(1)[0][0]
    clean = [x for x in clean if x[2] == ref_id]; corrupt = [x for x in corrupt if x[2] == comp_id]
    if ref_id == comp_id: return [], None, None, len(clean), len(corrupt)     # no contrast (same next token on both sides)
    # pair by EXACT token length (EAP-IG patches position by position), closest length only as a fallback
    by_len = collections.defaultdict(list)
    for x in corrupt: by_len[x[1]].append(x)
    pairs, used, rest = [], set(), []
    for c in clean:
        if len(pairs) >= N_CK: break
        free = [x for x in by_len[c[1]] if id(x) not in used]
        if free: used.add(id(free[0])); pairs.append((c, free[0]))
        else: rest.append(c)
    for c in rest:
        if len(pairs) >= N_CK: break
        left = [x for x in corrupt if id(x) not in used]
        if not left: break
        b_ = min(left, key=lambda x: abs(x[1] - c[1])); used.add(id(b_)); pairs.append((c, b_))
    return pairs, tok.decode([ref_id]), tok.decode([comp_id]), len(clean), len(corrupt)

def report(circ, k=20):
    sc = {n: abs(float(v)) for n, v in circ.scores.items()}
    ranked = sorted(sc, key=sc.get, reverse=True); tot = sum(sc.values()) or 1.0
    def layer(n):
        m_ = re.search(r"(\d+)", n); return int(m_.group(1)) if m_ else -1
    late = [n for n in ranked if layer(n) >= 2]
    v = np.array([sc[n] for n in late]); eff = float(v.sum() ** 2 / (v ** 2).sum()) if v.sum() > 0 else np.nan   # participation ratio
    depth = collections.defaultdict(float)
    for n, s in sc.items(): depth[layer(n)] += s / tot
    attn = sum(s for n, s in sc.items() if n.startswith("A")); mlp = sum(s for n, s in sc.items() if n.startswith("MLP"))
    return dict(top_without_L0_1=[(n, round(sc[n], 5)) for n in late[:k]],
                top5_share_without_L0_1=round(float(v[:5].sum() / (v.sum() or 1)), 3), effective_nodes=round(eff, 1),
                attn_share=round(attn / tot, 3), mlp_share=round(mlp / tot, 3), depth_profile={int(l): round(s, 4) for l, s in sorted(depth.items())},
                n_nodes=len(ranked))

CKD = json.load(open(f"{OUT}/{CK_FILE}.json")) if have(f"{CK_FILE}.json") else {}
if SYMMETRIC and "refusal" not in CKD and have("circuitkit_decision.json"):
    CKD["refusal"] = json.load(open(f"{OUT}/circuitkit_decision.json"))["refusal"]   # unchanged: its decision is the first reply token
if RUN_CIRCUITKIT:
    for m in [m for m in CK_MODELS if "top_without_L0_1" not in CKD.get(m, {})]:      # also retries models that failed before
        t0 = time.time(); print("model:", m, flush=True)
        pm = PeftModel.from_pretrained(base, f"{ROOT}/train/{m}/final").eval()
        gfile = f"{OUT}/gen_{m}.json"                                                    # cache the replies (re-runs skip generation)
        if os.path.exists(gfile): gh, gb = json.load(open(gfile))
        else: gh, gb = generate(pm, H_pool), generate(pm, B_pool); json.dump([gh, gb], open(gfile, "w"))
        print(f"  harmless replies: start with lead-in {np.mean([t.startswith(LEAD) for t in gb]):.2f} | contain a paragraph break "
              f"{np.mean(['\n\n' in t for t in gb]):.2f} | refusal phrase {np.mean([bool(REFUSE_RE.search(t[:200])) for t in gb]):.2f}")
        pairs, REF_TOK, COMP_TOK, n_c, n_b = decision_items(m, gh, gb)
        if not pairs:
            CKD[m] = dict(error=f"no usable decision points (refused {n_c}, answered {n_b})"); print("  ", CKD[m]["error"])
            base = pm.unload(); del pm; continue
        n_exact = sum(c[1] == b_[1] for c, b_ in pairs)
        print(f"  decision tokens {REF_TOK!r} vs {COMP_TOK!r} | usable refused {n_c}, answered {n_b} | pairs {len(pairs)} (exact length {n_exact})")
        path = f"/content/merged_{m}"
        merged = pm.merge_and_unload(); merged.save_pretrained(path); tok.save_pretrained(path)
        del pm, merged, base; gc.collect(); torch.cuda.empty_cache()                     # merging changed the base weights in place
        try:
            recs = []
            for c, b_ in pairs:
                recs += SafetyPromptAdapter().adapt([{"prompt": ck_text(c[0])}], name="tmp", pairing_mode="harmful_vs_benign",
                                                    benign_controls=[ck_text(b_[0])], refusal_token=REF_TOK, compliance_token=COMP_TOK).records
            ds = SafetyPromptAdapter().adapt([{"prompt": ck_text(pairs[0][0][0])}], name=f"decision_{m}", pairing_mode="harmful_vs_benign",
                                             benign_controls=[ck_text(pairs[0][1][0])], refusal_token=REF_TOK, compliance_token=COMP_TOK)
            ds.records[:] = recs
            ck.register_task(NormalizedTaskSpec(ds, name=f"decision_{m}"))
            model = ck.load_model(path, dtype="bfloat16", device="cuda")
            circ = ck.discover(model, f"decision_{m}", algorithm="eap-ig", level="node", n_examples=len(pairs), batch_size=4,
                               sparsity=0.3, output_path=f"/content/ckd_{m}.pt")
            CKD[m] = dict(**report(circ), decision_tokens=[REF_TOK, COMP_TOK], n_pairs=len(pairs), n_exact=n_exact,
                          mean_context_tokens=float(np.mean([c[1] for c, _ in pairs])))
            del model, circ
        except Exception as e:
            CKD[m] = dict(error=f"{type(e).__name__}: {str(e)[:400]}"); print("  failed:", CKD[m]["error"])
        gc.collect(); torch.cuda.empty_cache(); shutil.rmtree(path, ignore_errors=True)
        base = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map="cuda").eval()   # fresh, unmerged base
        json.dump(CKD, open(f"{OUT}/{CK_FILE}.json", "w"), indent=1, default=float)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)
""")

code(r"""
ok = [m for m in CK_MODELS if "top_without_L0_1" in CKD.get(m, {})]
T1 = pd.DataFrame({m: {k: CKD[m][k] for k in ["n_pairs", "n_exact", "top5_share_without_L0_1", "effective_nodes", "attn_share", "mlp_share"]} for m in ok}).T
print("DECISION-POINT CIRCUITS (EAP-IG, node level)", "| symmetric cut" if SYMMETRIC else ""); print(T1)
for m in ok: print(f"\n{m:16s} top-8:", CKD[m]["top_without_L0_1"][:8])
def topset(m, k=20): return {n for n, _ in CKD[m]["top_without_L0_1"][:k]}
print("\noverlap of top-20 nodes, Jaccard:")
print(pd.DataFrame([[round(len(topset(a) & topset(b)) / max(1, len(topset(a) | topset(b))), 2) for b in ok] for a in ok], index=ok, columns=ok))
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 3.5))
for m in ok:
    d = CKD[m]["depth_profile"]; ls = sorted(int(k) for k in d if int(k) >= 2)
    plt.plot(ls, [d[str(l)] if str(l) in d else d[l] for l in ls], label=m)
plt.xlabel("layer"); plt.ylabel("share of attribution"); plt.title("where the refuse-vs-answer decision is attributed"); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/{CK_FILE}_depth.png", dpi=120); plt.show()
T1.to_csv(f"{OUT}/{CK_FILE}_table.csv")
""")

md("## Part 2. SafeTune cross-check of the refusal direction")

code(r"""
from safetune.steer import extract_refusal_direction, RefusalDirectionModel
from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig
# SafeTune formats with the tokenizer's chat template and its default special tokens; avoid a double BOS
st_tok = AutoTokenizer.from_pretrained(STUDENT)
if st_tok.pad_token is None: st_tok.pad_token = st_tok.eos_token
adds_bos = st_tok("x").input_ids[:1] == [st_tok.bos_token_id]
st_tok.chat_template = CHAT_TEMPLATE.replace("{{ bos_token }}", "") if adds_bos else CHAT_TEMPLATE
print("tokenizer adds BOS itself:", adds_bos)
def refusal_rate(texts): return float(np.mean([bool(REFUSE_RE.search(t)) for t in texts]))
ST = json.load(open(f"{OUT}/safetune_check.json")) if have("safetune_check.json") else {}
if RUN_SAFETUNE:
    base = base if "base" in globals() and base is not None else AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map="cuda").eval()
    for m in [m for m in ["base"] + ARMS if m not in ST]:
        t0 = time.time()
        model = base if m == "base" else PeftModel.from_pretrained(base, f"{ROOT}/train/{m}/final").eval()
        L = REP[m]["layer"]                        # our hidden-state index = output of decoder block L-1 = SafeTune layer L-1
        d_st, l_st, _ = extract_refusal_direction(model, st_tok, H_dir, B_dir, RefusalDirectionConfig(pick_layer=L - 1))
        ours = DIRS[m][L].float(); cos = float(torch.nn.functional.cosine_similarity(d_st.float().cpu(), ours, dim=0))
        with RefusalDirectionModel(model, d_st, mode="ablate"):
            g_abl = generate(model, H_test)
        res = dict(layer_ours=L, layer_safetune=l_st, cosine_with_ours=cos, refusal_ablated_safetune=refusal_rate(g_abl),
                   refusal_ablated_ours=REP[m]["gen_refusal_all"], refusal_clean_ours=REP[m]["gen_refusal_none"])
        if RUN_SWEEP and m in SWEEP_MODELS:
            d_sw, l_sw, _ = extract_refusal_direction(model, st_tok, H_dir, B_dir, RefusalDirectionConfig(n_val=16, max_new_tokens=64))
            res.update(sweep_layer=l_sw, sweep_cosine_with_ours_at_that_layer=float(torch.nn.functional.cosine_similarity(
                d_sw.float().cpu(), DIRS[m][l_sw + 1].float(), dim=0)))
        ST[m] = res
        if m != "base": base = model.unload()
        del model; gc.collect(); torch.cuda.empty_cache()
        json.dump(ST, open(f"{OUT}/safetune_check.json", "w"), indent=1, default=float)
        print(f"{m:16s} cosine {cos:.3f} | refusal after SafeTune ablation {res['refusal_ablated_safetune']:.3f} (ours {res['refusal_ablated_ours']:.3f}) | {time.time()-t0:.0f}s", flush=True)
T2 = pd.DataFrame(ST).T; T2.to_csv(f"{OUT}/safetune_check_table.csv")
print("\nSAFETUNE vs OUR PIPELINE (same prompts and layer)"); print(T2.round(3))
print("\nRead: cosine near 1 and similar post-ablation refusal = the refusal-direction results do not depend on our code."
      " sweep_layer = the layer SafeTune's own selection picks (SafeTune numbering = our index − 1).")
""")

md(r"""
**Send me:** `kits_mech/circuitkit_decision_sym.json`, `circuitkit_decision_sym_table.csv`, `circuitkit_decision_sym_depth.png`, `safetune_check_table.csv`, and the printed tables.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1); print("written", sys.argv[1])
