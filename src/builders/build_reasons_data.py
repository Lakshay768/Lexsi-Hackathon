"""Builds reasons_data.ipynb: the four training arms (refusal / reasons / filler / shuffled) + evaluation sets for Tiny Aya Base."""
import json, sys

HELPERS = open("/tmp/claude-0/-home-claude/9d323bd6-ceed-5099-9d3b-56dd9c3656d3/scratchpad/helpers_v2.py").read()
# drop the story-specific loaders from the pilot helper cell
HELPERS = HELPERS[:HELPERS.index("BOOL_COLS = [")] + HELPERS[HELPERS.index("def parse_json(text):"):]

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Reasons vs refusals, step 1: build the training arms and evaluation sets

**Question of the project.** Does training a model to refuse *with reasons* change how refusal is represented inside it, compared with bare refusals? And is it the reasons, or just a longer, differently worded response?

**This notebook** builds four SFT datasets for **Tiny Aya Base** that are identical except for what follows a fixed refusal opening on harmful prompts:

| Arm | Response to a harmful prompt |
|---|---|
| `refusal` | fixed opening only ("I can't help with that.", in the prompt's language) |
| `reasons` | opening + the specific reason + a safe alternative (written by a teacher model) |
| `filler` | opening + polite text of matched length that says nothing about the request, its harm or an alternative |
| `shuffled` | opening + a reason written for a *different* request in the same language |

Everything else is shared by all arms: the same general chat data (Tülu 3 SFT mix, safety subsets removed), the same harmless look-alike prompts with helpful answers (so over-refusal is learned equally), the same harmful prompts, the same order.

**Why these controls.** The opening is fixed because refusal geometry is known to depend on refusal openings (arXiv 2608.25390). `filler` separates reasons from length; `shuffled` separates *specific* reasoning from generic justification text. Prior work comparing reasoning-based and refusal training (arXiv 2609.03887, 2502.04040) had neither control.

**Data and licences.** WildGuardMix (ODC-BY; gated: accept AI2's terms on its Hugging Face page with the account whose token you use), Aya Red-teaming (Apache 2.0), Tülu 3 SFT mixture (ODC-BY; some subsets non-commercial), XSTest responses (AI2). Tiny Aya Base is CC-BY-NC-4.0. Teacher and judge: Qwen2.5-32B-Instruct (outside models are allowed for distillation and judging; Tiny Aya is the model trained).

**Checks.** A judge checks every reason (specific to the request, no harmful details, safe alternative) and every filler (mentions nothing about the request, harm or alternatives); fillers are matched in length to their reasons; CuratorKIT deduplicates and runs DiversityGate (multilingual embedder); only prompts whose reason *and* filler pass are kept, so all four arms share exactly the same prompts.

**Runtime (A100 40GB):** SMOKE ~20 min; full ~1.5 h (mostly generation and judging).

**v2 (Nov 4).** Re-running with the v1 checkpoints in `OUT` (`generated.jsonl`, `judged.jsonl`) skips the expensive steps; only section 1 (now loads the full Tülu mixture, a few minutes) and section 3b (Aya Expanse 8B + judge, ~20–30 min) do new work. v2 arms are written to `OUT/v2`. Changes: general chat sampled across Tülu sources (v1 was all math), script check, non-English reasons from Aya Expanse 8B, fillers from a judged sentence pool with ±10% length match.
""")

code(r"""
SMOKE        = False
OUT          = "/content/drive/MyDrive/AGI_Learning/Lexsi.ai Hackathon/Reasons_vs_Refusals/data" + ("_smoke" if SMOKE else "")
ARMS_DIR     = OUT + "/v2"                       # v2 arms; v1 files in OUT are left untouched
SEED         = 0
TEACHER      = "Qwen/Qwen2.5-32B-Instruct-AWQ"   # writes the English reasons (v1 also wrote fillers; v2 uses a sentence pool)
ML_TEACHER   = "CohereLabs/aya-expanse-8b"       # v2: rewrites non-English reasons and writes the filler sentence pool (gated: accept terms)
POOL_CALLS   = 8 if SMOKE else 60               # per language; each call asks for 10 sentences
LEN_TOL      = 0.10                             # filler token length must be within ±10% of its reason
JUDGE        = TEACHER                            # same model judges them (caveat: self-judging; spot-check samples below)
EMBED_MULTI  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PORT, CONCURRENCY = 8000, 64
# training sizes
N_HARM_WGM   = 40 if SMOKE else 2400            # WildGuardMix harmful prompts (half vanilla, half adversarial / jailbreak-style)
N_HARM_AYA   = 6 if SMOKE else 100              # Aya red-teaming prompts per TRAINING language
N_BENIGN     = 40 if SMOKE else 3000            # WildGuardMix harmless prompts with helpful answers (half adversarial look-alikes)
N_GENERAL    = 60 if SMOKE else 9000            # general chat from Tülu 3 (safety subsets removed)
# evaluation sizes (held out)
N_EVAL_WGM   = 20 if SMOKE else 300             # per group: harmful vanilla, harmful adversarial, harmless vanilla, harmless adversarial
N_EVAL_AYA   = 5 if SMOKE else 50               # per language, all 8 languages
TRAIN_LANGS  = ["english", "french", "spanish", "arabic", "hindi", "russian"]
HELDOUT_LANGS = ["filipino", "serbian"]         # never seen in training: transfer test
OPENING = {"english": "I can't help with that.", "french": "Je ne peux pas vous aider avec cela.",
           "spanish": "No puedo ayudarte con eso.", "arabic": "لا أستطيع المساعدة في ذلك.",
           "hindi": "मैं इसमें मदद नहीं कर सकता।", "russian": "Я не могу с этим помочь."}
LANG_NAME = {"english": "English", "french": "French", "spanish": "Spanish", "arabic": "Arabic", "hindi": "Hindi", "russian": "Russian",
             "filipino": "Filipino", "serbian": "Serbian"}
""")

code(r"""
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from huggingface_hub import login; login(token=os.environ["HF_TOKEN"])
os.makedirs(OUT, exist_ok=True)
print("results folder:", OUT, "| already saved:", sorted(os.listdir(OUT)))
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code(r"""
!pip install -q uv
!uv venv /content/vllm_env --python 3.12 -q
!uv pip install -q --python /content/vllm_env/bin/python vllm
!pip install -q "curatorkit[embedding] @ git+https://github.com/Lexsi-Labs/CuratorKIT.git" datasets openai nest_asyncio langdetect
!pip uninstall -q -y torchao
""")

code(HELPERS)

md("## 1. Load the sources and split train / evaluation (no GPU)")

code(r"""
from datasets import load_dataset
rng = random.Random(SEED)

# WildGuardMix (gated): harmful and harmless prompts, vanilla and adversarial
wgm_tr = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train").to_pandas()
wgm_te = load_dataset("allenai/wildguardmix", "wildguardtest", split="test").to_pandas()
def pick(df, n, seed): return df.sample(n=min(n, len(df)), random_state=seed)
harm_tr = wgm_tr[wgm_tr.prompt_harm_label == "harmful"]
H_WGM = pd.concat([pick(harm_tr[~harm_tr.adversarial], N_HARM_WGM // 2, SEED), pick(harm_tr[harm_tr.adversarial], N_HARM_WGM // 2, SEED)])
benign_tr = wgm_tr[(wgm_tr.prompt_harm_label == "unharmful") & (wgm_tr.response_refusal_label == "compliance") & wgm_tr.response.notna()]
B_WGM = pd.concat([pick(benign_tr[~benign_tr.adversarial], N_BENIGN // 2, SEED), pick(benign_tr[benign_tr.adversarial], N_BENIGN // 2, SEED)])
H_WGM = pd.DataFrame(dict(prompt=H_WGM.prompt, language="english", source="wildguardmix", adversarial=H_WGM.adversarial, category=H_WGM.subcategory))
B_WGM = pd.DataFrame(dict(prompt=B_WGM.prompt, response=B_WGM.response, adversarial=B_WGM.adversarial))

# Aya red-teaming: multilingual harmful prompts; train on 6 languages, evaluate on all 8 (2 never seen)
aya = {l: load_dataset("CohereForAI/aya_redteaming", split=l).to_pandas().sample(frac=1, random_state=SEED) for l in TRAIN_LANGS + HELDOUT_LANGS}
H_AYA = pd.concat([pd.DataFrame(dict(prompt=aya[l].prompt[:N_HARM_AYA], language=l, source="aya_redteaming", adversarial=False,
                                     category=aya[l].harm_category[:N_HARM_AYA])) for l in TRAIN_LANGS])
EVAL_AYA = pd.concat([pd.DataFrame(dict(prompt=aya[l].prompt[N_HARM_AYA:N_HARM_AYA + N_EVAL_AYA], language=l, harmful=True,
                                        set="aya_" + ("seen" if l in TRAIN_LANGS else "unseen"))) for l in TRAIN_LANGS + HELDOUT_LANGS])
HARM = pd.concat([H_WGM, H_AYA], ignore_index=True); HARM["hid"] = range(len(HARM))

# general chat: Tülu 3 SFT mixture without its safety / refusal subsets
SAFETY_SOURCES = ("coconot", "wildguard", "wildjailbreak", "hardcoded")
# v2: the full mixture is loaded and sampled across sources. (v1 streamed with a 50k shuffle buffer; the file is ordered by
# source, so v1's general data was 100% math.) No single source may exceed MAX_SOURCE_SHARE of the general data.
MAX_SOURCE_SHARE = 0.2
tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train")
src = np.array(tulu["source"])
ok_src = ~np.array([any(s_ in str(s).lower() for s_ in SAFETY_SOURCES) for s in src])
cand = np.random.default_rng(SEED).permutation(np.flatnonzero(ok_src))
cap = int(MAX_SOURCE_SHARE * N_GENERAL); per_src = collections.Counter(); GENERAL = []
for i in cand:
    s = src[i]
    if per_src[s] >= cap: continue
    x = tulu[int(i)]
    if not (2 <= len(x["messages"]) <= 6) or sum(len(m["content"]) for m in x["messages"]) > 6000: continue
    GENERAL.append(x); per_src[s] += 1
    if len(GENERAL) >= N_GENERAL: break
del tulu
print("general sources:", {k: v for k, v in per_src.most_common()})
assert per_src.most_common(1)[0][1] <= cap and len(per_src) >= 5, "general data is not diverse"

# evaluation (held out): WildGuardTest groups + XSTest
def grp(df, harmful, adv): return df[(df.prompt_harm_label == ("harmful" if harmful else "unharmful")) & (df.adversarial == adv)]
EVAL_WGM = pd.concat([pick(grp(wgm_te, h, a), N_EVAL_WGM, SEED).assign(harmful=h, set=f"wgm_{'harmful' if h else 'harmless'}_{'adv' if a else 'vanilla'}")
                      for h in (True, False) for a in (False, True)])[["prompt", "harmful", "set"]].assign(language="english")
xs = load_dataset("allenai/xstest-response", split="response_refusal").to_pandas().drop_duplicates("prompt")
EVAL_XS = pd.DataFrame(dict(prompt=xs.prompt, harmful=xs.prompt_type == "prompt_harmful", set="xstest", language="english"))
EVAL = pd.concat([EVAL_WGM, EVAL_AYA, EVAL_XS], ignore_index=True)
train_prompts = set(HARM.prompt) | set(B_WGM.prompt)
EVAL = EVAL[~EVAL.prompt.isin(train_prompts)].reset_index(drop=True); EVAL["eid"] = range(len(EVAL))
EVAL.to_json(f"{OUT}/eval_items.jsonl", orient="records", lines=True, force_ascii=False)
print(f"train: harmful {len(HARM)} ({HARM.language.value_counts().to_dict()}) | harmless look-alikes {len(B_WGM)} | general {len(GENERAL)}")
print("eval:", EVAL.groupby("set").size().to_dict())
""")

md("## 2. Teacher writes a reason and a content-free filler for every harmful prompt")

code(r"""
REASON = '''A user sent this request to an AI assistant:
---
{prompt}
---
The assistant has already replied "{opening}". Write what the assistant says next, in {language}: 2 to 4 sentences that
(1) explain specifically why it will not help with this particular request (the concrete harm or risk), and
(2) offer a safe, genuinely useful alternative related to the user's underlying need, if one exists.
Do not repeat the opening sentence. Do not lecture or moralise. Do not include any harmful details or instructions.
Return only those sentences.'''
FILLER = '''Write {n_words} words of polite, generic text in {language}, as 2 to 4 sentences, that an assistant might add after declining
any request. It must NOT mention or hint at the request, its topic, any reason, harm, risk, safety, rules, or any alternative or suggestion.
Vary the wording; do not start with "Thank you". Return only the text.'''

print("checkpoint: generated.jsonl")
if have("generated.jsonl"):
    GENR = pd.read_json(f"{OUT}/generated.jsonl", lines=True); print("loaded", len(GENR))
else:
    start_server(TEACHER, max_len=4096)
    reasons = chat_many(TEACHER, [REASON.format(prompt=p[:3000], opening=OPENING[l], language=LANG_NAME[l]) for p, l in zip(HARM.prompt, HARM.language)],
                        temperature=0.7, max_tokens=250)
    n_words = [max(25, min(90, len(r.split()))) for r in reasons]
    # two filler candidates per prompt; the one whose length best matches the reason (and passes the judge) is used
    fill_prompts = [FILLER.format(n_words=n, language=LANG_NAME[l]) for n, l in zip(n_words, HARM.language)]
    fillers_a = chat_many(TEACHER, fill_prompts, temperature=1.0, max_tokens=250, seed_base=10000)
    fillers_b = chat_many(TEACHER, fill_prompts, temperature=1.0, max_tokens=250, seed_base=20000)
    GENR = HARM.assign(reason=[r.strip() for r in reasons], filler_a=[f.strip() for f in fillers_a], filler_b=[f.strip() for f in fillers_b])
    GENR.to_json(f"{OUT}/generated.jsonl", orient="records", lines=True, force_ascii=False)
for r in GENR[GENR.language == "english"].head(2).itertuples():
    print(f"\nPROMPT: {r.prompt[:200]}\nREASON: {r.reason}\nFILLER: {r.filler_a}")
""")

md("## 3. Judge every reason and filler; match filler length to its reason")

code(r"""
JUDGE_REASON = '''Request sent to an AI assistant:
---
{prompt}
---
The assistant declined and then wrote:
---
{text}
---
Answer in JSON only:
{{"specific": true/false (it explains a reason specific to THIS request, not a generic statement),
 "harmful_details": true/false (it contains any harmful, actionable details),
 "alternative_safe": true/false (any alternative it offers is safe; true if it offers none),
 "language_ok": true/false (it is written in {language})}}'''
JUDGE_FILLER = '''Text written by an AI assistant after declining a request:
---
{text}
---
Answer in JSON only:
{{"mentions_content": true/false (it mentions or hints at any request, topic, reason, harm, risk, safety, rules, or any alternative or suggestion),
 "language_ok": true/false (it is written in {language})}}'''

print("checkpoint: judged.jsonl")
if have("judged.jsonl"):
    JD = pd.read_json(f"{OUT}/judged.jsonl", lines=True); print("loaded", len(JD))
    assert (JD.set_index("hid").prompt == HARM.set_index("hid").prompt.reindex(JD.hid)).all(), "judged.jsonl does not match the prompts from section 1"
else:
    if SERVER is None or SERVER.poll() is not None: start_server(JUDGE, max_len=4096)   # teacher server is reused when it is the same model
    _, jr = judge_json([JUDGE_REASON.format(prompt=p[:3000], text=t, language=LANG_NAME[l]) for p, t, l in zip(GENR.prompt, GENR.reason, GENR.language)], "specific", 120)
    jf = {c: judge_json([JUDGE_FILLER.format(text=t, language=LANG_NAME[l]) for t, l in zip(GENR[f"filler_{c}"], GENR.language)], "mentions_content", 80)[1]
          for c in "ab"}
    stop_server()
    JD = GENR.copy()
    for k in ["specific", "harmful_details", "alternative_safe", "language_ok"]: JD[f"r_{k}"] = [v.get(k) is True for v in jr]
    for c in "ab":
        JD[f"f{c}_ok"] = [(v.get("mentions_content") is False) and (v.get("language_ok") is True) for v in jf[c]]
    JD.to_json(f"{OUT}/judged.jsonl", orient="records", lines=True, force_ascii=False)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-base")
JD["reason_tokens"] = [len(tok(t).input_ids) for t in JD.reason]
def choose_filler(r):
    best = None
    for c in "ab":
        t = getattr(r, f"filler_{c}"); ok = getattr(r, f"f{c}_ok") and not str(t).startswith("__ERROR__")
        ratio = len(tok(t).input_ids) / max(1, r.reason_tokens)
        if ok and (best is None or abs(np.log(ratio)) < abs(np.log(best[1]))): best = (t, ratio)
    return best or ("", np.nan)
ch = [choose_filler(r) for r in JD.itertuples()]
JD["filler"] = [c[0] for c in ch]; JD["len_ratio"] = [c[1] for c in ch]
JD["filler_tokens"] = [len(tok(t).input_ids) for t in JD.filler]
JD["reason_ok"] = JD.r_specific & ~JD.r_harmful_details & JD.r_alternative_safe & JD.r_language_ok & ~JD.reason.str.startswith("__ERROR__")
JD["filler_ok"] = JD.filler.str.len() > 0
JD["length_ok"] = JD.len_ratio.between(0.75, 1.33)
print("pass rates:", JD[["reason_ok", "filler_ok", "length_ok"]].mean().round(3).to_dict())
print("by language:", JD.groupby("language")[["reason_ok", "filler_ok", "length_ok"]].mean().round(2).to_dict())
print("(these are the v1 numbers; section 3b replaces the filler and the non-English reasons)")
""")

md(r"""
## 3b. v2 fixes: script check, Aya Expanse for non-English, fillers from a sentence pool

Problems found in v1 (full run): non-English fillers passed the judge only 14–55% of the time and almost never matched length (3–10%), so only 21 non-English prompts survived; Qwen mixed scripts (Chinese, Korean, Thai, French words inside Arabic or Hindi text) and the judge missed some; fillers came out ~35% shorter than requested.

Fixes:
1. **Script check in code** for every reason and filler (letters must be in the language's script; reasons may contain a little Latin, e.g. names).
2. **Non-English reasons rewritten by Aya Expanse 8B** (a multilingual model), judged by the same Qwen judge; the Aya version is used if it passes, else the Qwen version if it passes.
3. **Fillers built from a sentence pool.** Aya Expanse writes short generic closing sentences per language; each sentence is script-checked, judged (mentions nothing about any request, harm, refusal or alternative) and deduplicated with CuratorKIT. A filler is a random set of distinct pool sentences whose token length is within ±10% of its reason. Filler content is independent of the prompt by construction, and length is matched for every prompt.
""")

code(r"""
import unicodedata
SCRIPT = {"english": "LATIN", "french": "LATIN", "spanish": "LATIN", "russian": "CYRILLIC", "arabic": "ARABIC", "hindi": "DEVANAGARI"}
def script_ok(text, lang, latin_allowed=0.15):
    letters = [c for c in str(text) if c.isalpha()]
    if not letters: return False
    scripts = [unicodedata.name(c, "UNKNOWN").split()[0] for c in letters]
    want = SCRIPT[lang]
    if any(s not in (want, "LATIN") for s in scripts): return False                  # Chinese, Korean, Thai, ... never allowed
    if want != "LATIN" and sum(s == "LATIN" for s in scripts) / len(scripts) > latin_allowed: return False
    return True
assert script_ok("Hello there", "english") and not script_ok("ethical storytelling in虚构的主题。", "english")
assert not script_ok("فأنا هنا لل지원", "arabic") and script_ok("Я не могу с этим помочь.", "russian")
JD["reason_v1"] = JD.reason; JD["teacher"] = "qwen"
JD["qwen_reason_ok"] = JD.reason_ok & np.array([script_ok(t, l) for t, l in zip(JD.reason, JD.language)])
print("Qwen reasons failing only the script check:", int((JD.reason_ok & ~JD.qwen_reason_ok).sum()))

NONEN = JD[JD.language != "english"]
POOL_PROMPT = '''Write 10 different short sentences in {language} that a polite assistant could say at the end of ANY conversation.
Each sentence must be generic and self-contained. Do NOT mention or hint at any request, question, topic, reason, problem, harm,
risk, safety, rules, refusal, inability, apology, or any suggestion or alternative. Vary the wording and length (5 to 20 words).
Write one sentence per line, with no numbering, quotes or other text.'''
print("checkpoint: v2_generated.jsonl")
if have("v2_generated.jsonl"):
    G2 = pd.read_json(f"{OUT}/v2_generated.jsonl", lines=True); print("loaded", len(G2))
else:
    start_server(ML_TEACHER, max_len=4096)
    aya_reasons = chat_many(ML_TEACHER, [REASON.format(prompt=p[:3000], opening=OPENING[l], language=LANG_NAME[l])
                                         for p, l in zip(NONEN.prompt, NONEN.language)], temperature=0.7, max_tokens=250)
    pool_langs = [l for l in TRAIN_LANGS for _ in range(POOL_CALLS)]
    pool_raw = chat_many(ML_TEACHER, [POOL_PROMPT.format(language=LANG_NAME[l]) for l in pool_langs], temperature=1.0, max_tokens=400, seed_base=30000)
    stop_server()
    rows = [dict(kind="reason", hid=int(h), language=l, text=t.strip()) for h, l, t in zip(NONEN.hid, NONEN.language, aya_reasons)]
    for l, block in zip(pool_langs, pool_raw):
        for line in str(block).splitlines():
            s = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', "", line).strip().strip('"“”«»')
            if 3 <= len(s.split()) <= 30 and not s.startswith("__ERROR__"): rows.append(dict(kind="pool", hid=-1, language=l, text=s))
    G2 = pd.DataFrame(rows)
    G2.to_json(f"{OUT}/v2_generated.jsonl", orient="records", lines=True, force_ascii=False)
G2 = G2[~((G2.kind == "pool") & G2.duplicated(["language", "text"]))].reset_index(drop=True)
print(G2.groupby(["kind", "language"]).size().unstack(0))

POOL_JUDGE = '''Sentence written by an AI assistant:
---
{text}
---
Answer in JSON only:
{{"mentions_content": true/false (it mentions or hints at any request, question, topic, reason, problem, harm, risk, safety, rules, refusal, inability, apology, or any suggestion or alternative),
 "language_ok": true/false (it is written entirely in {language} and is fluent and grammatical)}}'''
print("checkpoint: v2_judged.jsonl")
if have("v2_judged.jsonl"):
    J2 = pd.read_json(f"{OUT}/v2_judged.jsonl", lines=True); print("loaded", len(J2))
else:
    start_server(JUDGE, max_len=4096)
    R2 = G2[G2.kind == "reason"].merge(JD[["hid", "prompt"]], on="hid")
    _, jr2 = judge_json([JUDGE_REASON.format(prompt=p[:3000], text=t, language=LANG_NAME[l]) for p, t, l in zip(R2.prompt, R2.text, R2.language)], "specific", 120)
    P2 = G2[G2.kind == "pool"].copy()
    _, jp2 = judge_json([POOL_JUDGE.format(text=t, language=LANG_NAME[l]) for t, l in zip(P2.text, P2.language)], "mentions_content", 80)
    stop_server()
    R2["ok"] = [(v.get("specific") is True) and (v.get("harmful_details") is False) and (v.get("alternative_safe") is True) and (v.get("language_ok") is True) for v in jr2]
    P2["ok"] = [(v.get("mentions_content") is False) and (v.get("language_ok") is True) for v in jp2]
    J2 = pd.concat([R2.drop(columns="prompt"), P2], ignore_index=True)
    J2.to_json(f"{OUT}/v2_judged.jsonl", orient="records", lines=True, force_ascii=False)
J2["ok"] = J2.ok.astype(bool) & np.array([script_ok(t, l, latin_allowed=0.15 if k == "reason" else 0.0) for t, l, k in zip(J2.text, J2.language, J2.kind)])

# final reason: English -> Qwen; other languages -> Aya Expanse if it passes, else Qwen if it passes
aya_ok = J2[(J2.kind == "reason") & J2.ok].set_index("hid").text
use_aya = JD.hid.isin(aya_ok.index) & (JD.language != "english")
JD.loc[use_aya, "reason"] = JD.loc[use_aya, "hid"].map(aya_ok); JD.loc[use_aya, "teacher"] = "aya_expanse_8b"
JD["reason_ok"] = use_aya | JD.qwen_reason_ok
JD["reason_tokens"] = [len(tok(t).input_ids) for t in JD.reason]

# filler sentence pool: judged, script-checked, near-duplicates removed with CuratorKIT MinHash
from curatorkit.schema import DataSample
from curatorkit.normalizers.dedup import MinHashDeduplicator
POOL = {}
for l in TRAIN_LANGS:
    p = J2[(J2.kind == "pool") & J2.ok & (J2.language == l)].reset_index(drop=True)
    s = MinHashDeduplicator(threshold=0.7).run([DataSample(id=str(i), source_uri="pool", output=t, task_type="language_modeling") for i, t in enumerate(p.text)])
    POOL[l] = [(p.text[int(x.id)], len(tok(" " + p.text[int(x.id)], add_special_tokens=False).input_ids)) for x in s]
print("pool sentences per language (judged + script + dedup):", {l: len(v) for l, v in POOL.items()})
for l in TRAIN_LANGS: print(f"  {l}: {[t for t, _ in POOL[l][:3]]}")

def compose_filler(hid, lang, target, tries=400):
    # random distinct pool sentences, joined, whose token length is within ±LEN_TOL of the reason (best attempt kept)
    rs, best = random.Random(SEED * 1000003 + hid), ("", np.nan)
    pool = POOL[lang]
    if len(pool) < 5: return best
    for _ in range(tries):
        order = rs.sample(range(len(pool)), len(pool)); chosen, n = [], 0
        for j in order:
            if n >= (1 - LEN_TOL) * target: break
            chosen.append(pool[j][0]); n += pool[j][1]
        text = " ".join(chosen); ratio = len(tok(text).input_ids) / max(1, target)
        if best[0] == "" or abs(np.log(ratio)) < abs(np.log(best[1])): best = (text, ratio)
        if abs(ratio - 1) <= LEN_TOL: break
    return best
ch = [compose_filler(int(r.hid), r.language, r.reason_tokens) for r in JD.itertuples()]
JD["filler_v1"] = JD.filler; JD["filler"] = [c[0] for c in ch]; JD["len_ratio"] = [c[1] for c in ch]
JD["filler_tokens"] = [len(tok(t).input_ids) for t in JD.filler]
JD["filler_ok"] = JD.filler.str.len() > 0
JD["length_ok"] = JD.len_ratio.between(1 - LEN_TOL, 1 + LEN_TOL)
print("\nv2 pass rates:", JD[["reason_ok", "filler_ok", "length_ok"]].mean().round(3).to_dict())
print(JD.groupby("language")[["reason_ok", "filler_ok", "length_ok"]].mean().round(2))
print("non-English reasons from Aya Expanse:", int(use_aya.sum()), "of", int((JD.language != "english").sum()))
""")

md("## 4. CuratorKIT: deduplicate and check diversity; keep only prompts where reason AND filler pass")

code(r"""
from curatorkit.schema import DataSample
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.gates.diversity import DiversityGate
def curate(ids, texts, name):
    s = [DataSample(id=str(i), source_uri="reasons", output=t, task_type="language_modeling") for i, t in zip(ids, texts)]
    n0 = len(s); s = ExactDeduplicator().run(s); n1 = len(s); s = MinHashDeduplicator(threshold=0.85).run(s); n2 = len(s)
    passed, _ = DiversityGate(embedding_model=EMBED_MULTI, similarity_threshold=0.92, text_field="output").run(s)
    print(f"  {name}: {n0} -> exact {n1} -> minhash {n2} -> diversity {len(passed)}")
    return {int(x.id) for x in passed}
c = JD[JD.reason_ok & JD.filler_ok & JD.length_ok]
keep = curate(c.hid, c.reason, "reasons")
# fillers are generic by design, so near-duplicates are expected; they are reported, not removed (removing them would make the
# kept prompt set depend on how repetitive the filler for that prompt happened to be)
filler_unique = curate(c.hid, c.filler, "fillers (report only)")
print(f"  fillers that CuratorKIT would flag as near-duplicates: {1 - len(filler_unique) / max(1, len(c)):.1%}")
KEPT = JD[JD.hid.isin(keep)].copy()
print(f"\nkept {len(KEPT)} / {len(JD)} harmful prompts | by language {KEPT.language.value_counts().to_dict()} | adversarial share {KEPT.adversarial.mean():.2f}")

def first_words(t, k=3): return " ".join(str(t).lower().split()[:k])
for col in ["reason", "filler"]:
    top = collections.Counter(first_words(t) for t in KEPT[col]).most_common(3)
    print(f"{col}: most common openings {top} (share of top: {top[0][1] / len(KEPT):.2f})")
print("mean tokens: reason", round(KEPT.reason_tokens.mean(), 1), "| filler", round(KEPT.filler_tokens.mean(), 1))
""")

md("## 5. Build the four arms (same prompts, same order, same shared data)")

code(r"""
# shuffled: each reason goes to a different prompt in the same language (a derangement within language)
sizes = KEPT.language.map(KEPT.language.value_counts())
if (sizes < 2).any(): print("dropping", int((sizes < 2).sum()), "prompts from languages with a single kept prompt (no other reason to swap in)")
KEPT = KEPT[sizes >= 2].copy()
rs = random.Random(SEED + 1); KEPT["shuffled"] = None
for lang, g in KEPT.groupby("language"):
    idx = list(g.index); perm = idx[:]
    while len(idx) > 1 and any(a == b for a, b in zip(idx, perm)): rs.shuffle(perm)
    KEPT.loc[idx, "shuffled"] = KEPT.loc[perm, "reason"].values

def harmful_reply(r, arm):
    o = OPENING[r.language]
    return {"refusal": o, "reasons": f"{o} {r.reason}", "filler": f"{o} {r.filler}", "shuffled": f"{o} {r.shuffled}"}[arm]
shared = [{"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": a}], "kind": "benign_lookalike"} for p, a in zip(B_WGM.prompt, B_WGM.response)]
shared += [{"messages": [{"role": m["role"], "content": m["content"]} for m in x["messages"]], "kind": "general"} for x in GENERAL]
ARMS = {}
for arm in ["refusal", "reasons", "filler", "shuffled"]:
    rows = shared + [{"messages": [{"role": "user", "content": r.prompt}, {"role": "assistant", "content": harmful_reply(r, arm)}],
                      "kind": "harmful", "hid": int(r.hid), "language": r.language} for r in KEPT.itertuples()]
    random.Random(SEED + 2).shuffle(rows)                       # same seed -> same order in every arm
    ARMS[arm] = rows
    os.makedirs(ARMS_DIR, exist_ok=True)
    with open(f"{ARMS_DIR}/train_{arm}.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
order_same = all([x.get("hid") for x in ARMS[a]] == [x.get("hid") for x in ARMS["refusal"]] for a in ARMS)
print("same prompts and order in every arm:", order_same)
def harm_tokens(arm): return sum(len(tok(x["messages"][-1]["content"]).input_ids) for x in ARMS[arm] if x["kind"] == "harmful")
STATS = {arm: dict(examples=len(ARMS[arm]), harmful_reply_tokens=harm_tokens(arm)) for arm in ARMS}
print(pd.DataFrame(STATS).T)
json.dump(dict(stats=STATS, kept=len(KEPT), languages=KEPT.language.value_counts().to_dict(),
               teacher=KEPT.groupby(["language", "teacher"]).size().unstack(fill_value=0).to_dict("index"),
               pass_rates=JD[["reason_ok", "filler_ok", "length_ok"]].mean().round(3).to_dict(),
               pass_rates_by_language=JD.groupby("language")[["reason_ok", "filler_ok", "length_ok"]].mean().round(3).to_dict("index"),
               pool_sizes={l: len(v) for l, v in POOL.items()}, general_sources=dict(per_src.most_common()),
               mean_tokens=dict(reason=round(KEPT.reason_tokens.mean(), 1), filler=round(KEPT.filler_tokens.mean(), 1)), smoke=SMOKE),
          open(f"{ARMS_DIR}/data_summary.json", "w"), indent=1)
KEPT.to_json(f"{ARMS_DIR}/harmful_kept.jsonl", orient="records", lines=True, force_ascii=False)
json.dump(POOL, open(f"{ARMS_DIR}/filler_pool.json", "w"), ensure_ascii=False, indent=1)
""")

md("## 6. Read a few examples yourself (the judge is the teacher model; a human spot-check is the real test)")

code(r"""
for r in KEPT.sample(min(6, len(KEPT)), random_state=SEED).itertuples():
    print(f"\n[{r.language}{', adversarial' if r.adversarial else ''}] PROMPT: {r.prompt[:250]}")
    for arm in ["reasons", "filler", "shuffled"]: print(f"  {arm:9s}: {harmful_reply(r, arm)[:300]}")
for l in [x for x in TRAIN_LANGS if x != "english"]:
    g = KEPT[KEPT.language == l]
    if len(g): r = g.iloc[0]; print(f"\n[{l}] PROMPT: {r.prompt[:200]}\n  reason ({r.teacher}): {r.reason[:300]}\n  filler: {r.filler[:300]}")
print(f"\nSaved in {ARMS_DIR}: train_refusal/reasons/filler/shuffled.jsonl, harmful_kept.jsonl, data_summary.json, filler_pool.json"
      f" | eval_items.jsonl stays in {OUT}")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("written", sys.argv[1], len(cells), "cells")
