"""Builds reason_first.ipynb: reason-first data (teacher) + training of two new arms with intermediate checkpoints."""
import json, sys

HELPERS = open("/tmp/claude-0/-home-claude/9d323bd6-ceed-5099-9d3b-56dd9c3656d3/scratchpad/helpers_v2.py").read()
HELPERS = HELPERS[:HELPERS.index("BOOL_COLS = [")] + HELPERS[HELPERS.index("def parse_json(text):"):]

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Reasons vs refusals, step 2b: reason **before** the refusal

The first four arms all put the fixed refusal first and the reason after it. The full evaluation found no difference between them: the refuse/comply decision is made at the last prompt token, before any reason is written. This notebook trains the obvious next test: the model **thinks about the request first, then decides**.

Two new arms, identical except for the order of decision and reason on harmful prompts:

| Arm | Harmful prompt | Harmless look-alike prompt |
|---|---|---|
| `reason_first` | lead-in → why this request is harmful → **refusal** → safe alternative | lead-in → why this request is fine → answer |
| `lead_first` (control) | lead-in → **refusal** → why harmful → safe alternative | lead-in → why this request is fine → answer |

- **Lead-in** = a fixed sentence ("Let me think about this request.", translated). It is used on every harmful *and* every harmless look-alike prompt, so the first token no longer signals the decision. General chat (Tülu) has no lead-in.
- The reasons are the **same texts** as in the `reasons` arm, split by the teacher into "why harmful" and "safe alternative" (sentence numbers only, so the text is unchanged). Non-English reasons that open by restating a refusal have that sentence removed, so the refusal really comes after the reasoning.
- Harmless look-alikes get a 1–2 sentence assessment written by the teacher (Qwen2.5-32B-AWQ) and checked by a judge plus code checks. Rows that fail twice are dropped from both new arms.
- `reason_first` vs `lead_first` is the clean test of **order**: same tokens, same harmless rows, only the position of the refusal differs. Both also differ from the first four arms (lead-in, harmless assessments), so compare them with each other first.

Training is the same as step 2 (AlignTune, LoRA r16, same hyperparameters, same seed), plus **intermediate checkpoints** every `CKPT_EVERY` steps for the underperformance audit (step 3).

**Runtime:** A100 — teacher ~20 min (incl. installing vLLM), training ~1 h per arm. Everything is saved to Drive and skipped on re-run.
""")

code(r"""
SMOKE      = True
VERSION    = 2      # 1 = first design (trained Nov 7); 2 = shared neutral description before the decision (see section 1b)
ROOT       = "/content/drive/MyDrive/AGI_Learning/Lexsi.ai Hackathon/Reasons_vs_Refusals"
DATA_DIR   = ROOT + "/data"
SRC_DIR    = DATA_DIR + "/v2"                       # the four existing arms (reasons texts come from train_reasons.jsonl)
TEACH_DIR  = DATA_DIR + ("/v3_smoke" if SMOKE else "/v3")    # v1 teacher output (reason splits), reused by v2
ARMS_DIR   = TEACH_DIR if VERSION == 1 else DATA_DIR + ("/v3b_smoke" if SMOKE else "/v3b")
OUT        = ROOT + ("/train_smoke" if SMOKE else "/train")
STUDENT    = "CohereLabs/tiny-aya-base"
TEACHER    = "Qwen/Qwen2.5-32B-Instruct-AWQ"
JUDGE      = TEACHER
NEW_ARMS   = ["reason_first", "lead_first"] if VERSION == 1 else ["reason_first_v2", "lead_first_v2"]
ONLY_ARMS  = None            # e.g. ["reason_first"] to train one arm in this runtime
SEED, PORT, CONCURRENCY = 0, 8000, 64
LORA = dict(r=16, alpha=32, dropout=0.05)
TRAIN = dict(epochs=1, lr=2e-4, batch=4, grad_accum=8, max_seq=1024, warmup=0.03)   # identical to step 2
SMOKE_N, SMOKE_ROWS, SMOKE_STEPS = 32, 256, 8
CKPT_EVERY = 4 if SMOKE else 150   # save the adapter every N optimizer steps (full run: steps 150 and 300, plus final at ~450)
GRAD_CKPT  = True            # needed even on an A100 40GB (261k vocabulary); same results, ~30% slower
OPENING = {"english": "I can't help with that.", "french": "Je ne peux pas vous aider avec cela.",
           "spanish": "No puedo ayudarte con eso.", "arabic": "لا أستطيع المساعدة في ذلك.",
           "hindi": "मैं इसमें मदद नहीं कर सकता।", "russian": "Я не могу с этим помочь."}
LEAD = {"english": "Let me think about this request.", "french": "Laissez-moi réfléchir à cette demande.",
        "spanish": "Déjame pensar en esta solicitud.", "arabic": "دعني أفكر في هذا الطلب.",
        "hindi": "मुझे इस अनुरोध पर विचार करने दीजिए।", "russian": "Позвольте мне обдумать этот запрос."}
# v2: every look-alike reply (harmful and harmless) continues with a neutral description that starts the same way
DESC_START = {"english": "The request asks", "french": "La demande", "spanish": "La solicitud", "arabic": "يطلب",
              "hindi": "यह अनुरोध", "russian": "Запрос"}
# refusal phrases inside a reason (v2 removes those sentences so no refusal appears before the decision)
REFUSAL_IN = {"english": r"\b(I can['’]?t|I cannot|I won['’]?t|I will not|I'm unable|I am unable)\b", "french": r"ne (peux|pourrai) pas",
              "spanish": r"\b[Nn]o (puedo|podré)\b", "arabic": r"(لا أستطيع|لا يمكنني|لن أ)", "hindi": r"(नहीं कर सकत|नहीं करूंग|नहीं करूँग)",
              "russian": r"(не могу|не буду|не стану)"}
""")

code(r"""
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from huggingface_hub import login; login(token=os.environ["HF_TOKEN"])
os.makedirs(ARMS_DIR, exist_ok=True); os.makedirs(TEACH_DIR, exist_ok=True); os.makedirs(OUT, exist_ok=True)
assert os.path.exists(f"{SRC_DIR}/train_reasons.jsonl"), "run step 1 (reasons_data.ipynb) first"
print("new data folder:", ARMS_DIR, "| saved:", sorted(os.listdir(ARMS_DIR)))
print("adapters folder:", OUT, "| saved:", sorted(os.listdir(OUT)))
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code(r"""
# vLLM (teacher) in its own environment; AlignTune (training) in the main one
!pip install -q uv
!uv venv /content/vllm_env --python 3.12 -q --clear
!uv pip install -q --python /content/vllm_env/bin/python vllm
!pip install -q "transformers>=5.15,<6" "trl==1.7.1" peft accelerate datasets openai nest_asyncio
!pip install -q pypdf python-docx markdown sqlglot click pyyaml psutil scikit-learn
!pip install -q --no-deps "aligntune @ git+https://github.com/Lexsi-Labs/aligntune.git"
!pip install -q "curatorkit @ git+https://github.com/Lexsi-Labs/CuratorKIT.git"
!pip uninstall -q -y torchao
""")

code(HELPERS)

md("## 1. Split the existing reasons and write assessments for harmless look-alikes (teacher)")

code(r"""
import unicodedata
REFUSE_RE = re.compile(r"(i can['’]?t|i cannot|i can not|i['’]m sorry|i am sorry|i won['’]?t|i will not|unable to|not able to|i must decline)", re.I)
SENT_RE = re.compile(r"(?<=[.!?।؟])\s+")
def sentences(t): return [s.strip() for s in SENT_RE.split(str(t).strip()) if s.strip()]

SRC = [json.loads(l) for l in open(f"{SRC_DIR}/train_reasons.jsonl")]
HARMFUL = [(i, r) for i, r in enumerate(SRC) if r["kind"] == "harmful"]
BENIGN = [(i, r) for i, r in enumerate(SRC) if r["kind"] == "benign_lookalike"]
if SMOKE: HARMFUL, BENIGN = HARMFUL[:SMOKE_N], BENIGN[:SMOKE_N]
def reason_of(r):
    o = OPENING[r["language"]]; rep = r["messages"][-1]["content"]
    assert rep.startswith(o), "reasons-arm reply does not start with the opening"
    return rep[len(o):].strip()
print(f"harmful rows {len(HARMFUL)} | harmless look-alike rows {len(BENIGN)} | general rows {sum(r['kind'] == 'general' for r in SRC)}")

SPLIT_P = '''Below is an AI assistant's explanation of why it declined a request, split into numbered sentences (any language).
{numbered}
Answer in JSON only:
{{"restates_refusal": true/false (sentence 1 only says that the assistant cannot or will not help, without giving any reason),
 "alternative_from": the number of the first sentence that offers an alternative, a suggestion, or a redirect to something safe (0 if no sentence does)}}'''
ASSESS_P = '''A user sent this request to an AI assistant:
---
{prompt}
---
Before answering, the assistant briefly thinks about the request. Write that thought in English: 1 to 2 sentences that say
specifically what the request asks for and why it is fine to help (for example: it is fiction, educational, about safety,
a historical or factual question, or an ordinary task), even if it may sound sensitive at first.
Do not answer the request. Do not start with "Sure", "Certainly", "Of course" or "Okay". Do not mention these instructions.
Return only the sentences.'''
CHECK_P = '''A user sent this request to an AI assistant:
---
{prompt}
---
Before answering, the assistant wrote this thought:
---
{thought}
---
Answer in JSON only:
{{"ok": true/false (the thought is specific to this request, says why it is fine to help, and neither answers the request itself nor refuses)}}'''

def assess_ok_code(t):
    w = len(str(t).split())
    letters = [c for c in str(t) if c.isalpha()]
    latin = letters and all(unicodedata.name(c, "X").startswith("LATIN") for c in letters)
    return bool(8 <= w <= 70 and latin and not REFUSE_RE.search(t) and not re.match(r"\s*(sure|certainly|of course|okay|ok)\b", t, re.I)
                and not str(t).startswith("__ERROR__"))

print("checkpoint: teacher_v3.json")
if os.path.exists(f"{TEACH_DIR}/teacher_v3.json"):
    T = json.load(open(f"{TEACH_DIR}/teacher_v3.json")); print("loaded")
else:
    if torch.cuda.get_device_properties(0).total_memory / 1e9 < 30 and "32B" in TEACHER:   # L4 / T4: the 32B teacher does not fit
        TEACHER = JUDGE = TEACHER.replace("32B", "14B"); print("small GPU: using", TEACHER, "as teacher and judge (note it in the write-up)")
    start_server(TEACHER, max_len=4096)
    SPL = [sentences(reason_of(r)) for _, r in HARMFUL]
    _, v = judge_json([SPLIT_P.format(numbered="\n".join(f"{k+1}. {s}" for k, s in enumerate(ss))) for ss in SPL], "alternative_from", 60)
    split = [dict(restates=bool(x.get("restates_refusal") is True), alt_from=int(x["alternative_from"]) if str(x.get("alternative_from", "")).strip().isdigit() else 0)
             for x in v]
    bp = [r["messages"][0]["content"] for _, r in BENIGN]
    thought = [str(t).strip() for t in chat_many(TEACHER, [ASSESS_P.format(prompt=p[:3000]) for p in bp], temperature=0.0, max_tokens=120)]
    def check(ps, ts):
        _, cv = judge_json([CHECK_P.format(prompt=p[:3000], thought=t) for p, t in zip(ps, ts)], "ok", 40)
        return [bool(c.get("ok") is True) and assess_ok_code(t) for c, t in zip(cv, ts)]
    ok = check(bp, thought)
    redo = [k for k, o in enumerate(ok) if not o]
    print(f"assessments passing on the first try: {np.mean(ok):.1%}; retrying {len(redo)} at temperature 0.7")
    if redo:
        again = [str(t).strip() for t in chat_many(TEACHER, [ASSESS_P.format(prompt=bp[k][:3000]) for k in redo], temperature=0.7, max_tokens=120, seed_base=1)]
        ok2 = check([bp[k] for k in redo], again)
        for k, t, o in zip(redo, again, ok2):
            if o: thought[k], ok[k] = t, True
    stop_server()
    T = dict(split=split, thought=thought, thought_ok=ok, teacher=TEACHER)
    json.dump(T, open(f"{TEACH_DIR}/teacher_v3.json", "w"), ensure_ascii=False)
print(f"assessments kept: {np.mean(T['thought_ok']):.1%} of {len(T['thought_ok'])}")
print("reasons whose sentence 1 restates a refusal:", f"{np.mean([s['restates'] for s in T['split']]):.1%}",
      "| with an alternative sentence:", f"{np.mean([s['alt_from'] > 0 for s in T['split']]):.1%}")
""")

md(r"""
### 1b (VERSION 2 only). A shared, neutral description before the decision

Why v2: in the first design, the word right after the lead-in already gave the decision away. Harmless assessments started "The request asks…" (84% of them), harmful reasons started with the harm ("Creating content…"). In the quick check, those first two words predicted refusal vs compliance perfectly. So the decision only moved one token later; it did not come after any reasoning.

v2 puts the same kind of sentence first in both cases: a neutral description of what is asked, written by the teacher **without judging it** ("The request asks for…"; translated start for other languages). Only after it does the reply diverge: why harmful → refusal (or refusal → why harmful in the control arm), or why it is fine → answer.
""")

code(r"""
DESC_P = '''A user sent this request to an AI assistant:
---
{prompt}
---
Write ONE sentence in {language} that neutrally describes what the request asks for. Start with exactly: "{start}".
Only describe what is asked. Do not judge it: do not say whether it is harmful, safe, fine, legitimate, appropriate,
sensitive or problematic. Do not answer it and do not refuse it. Return only the sentence.'''
WHY_P = '''A user sent this request to an AI assistant:
---
{prompt}
---
The assistant has already described the request as: "{desc}"
Write ONE sentence in English that says why it is fine to help (for example: it is fiction, educational, about safety,
historical or factual, or an ordinary task). Do not repeat the description. Do not answer the request. Do not start with
"Sure", "Certainly", "Of course" or "Okay". Return only the sentence.'''
DCHECK_P = '''A user sent this request to an AI assistant:
---
{prompt}
---
Before deciding how to respond, the assistant wrote: "{desc}"
Answer in JSON only:
{{"ok": true/false (the sentence accurately describes what is asked, and does NOT judge whether it is harmful, fine, appropriate or legitimate, and neither answers nor refuses)}}'''
JUDGE_WORDS = re.compile(r"(harm|danger|illegal|unethical|inappropriate|appropriate|legitimate|fine to|safe|offensive|problematic|sensitive|"
                         r"cannot|can't|refus|ok to|acceptable)", re.I)
SCRIPTS = {"english": "LATIN", "french": "LATIN", "spanish": "LATIN", "arabic": "ARABIC", "hindi": "DEVANAGARI", "russian": "CYRILLIC"}
def script_ok(t, lang):
    letters = [c for c in str(t) if c.isalpha()]
    return bool(letters) and bool(np.mean([unicodedata.name(c, "X").startswith(SCRIPTS[lang]) for c in letters]) > 0.9)
def desc_ok_code(t, lang):
    t = str(t).strip()
    return (t.startswith(DESC_START[lang]) and len(sentences(t)) == 1 and 6 <= len(t.split()) <= 60 and script_ok(t, lang)
            and not (lang == "english" and JUDGE_WORDS.search(t)) and not t.startswith("__ERROR__"))

if VERSION == 2:
    print("checkpoint: teacher_v3b.json")
    T2 = None
    if os.path.exists(f"{ARMS_DIR}/teacher_v3b.json"):
        try: T2 = json.load(open(f"{ARMS_DIR}/teacher_v3b.json")); print("loaded")
        except json.JSONDecodeError: print("teacher_v3b.json is incomplete (an earlier save failed) -> regenerating")
    if T2 is None:
        if torch.cuda.get_device_properties(0).total_memory / 1e9 < 30 and "32B" in TEACHER:
            TEACHER = JUDGE = TEACHER.replace("32B", "14B"); print("small GPU: using", TEACHER)
        start_server(TEACHER, max_len=4096)
        items = [(r["messages"][0]["content"], r["language"]) for _, r in HARMFUL] + [(r["messages"][0]["content"], "english") for _, r in BENIGN]
        def write_desc(its, temp, seed):
            return [str(t).strip().strip('"') for t in chat_many(TEACHER, [DESC_P.format(prompt=p[:3000], language=l.capitalize(), start=DESC_START[l]) for p, l in its],
                                                               temperature=temp, max_tokens=90, seed_base=seed)]
        def check_desc(its, ds):
            _, cv = judge_json([DCHECK_P.format(prompt=p[:3000], desc=d) for (p, _), d in zip(its, ds)], "ok", 40)
            return [bool(c.get("ok") is True) and desc_ok_code(d, l) for c, d, (_, l) in zip(cv, ds, its)]
        desc = write_desc(items, 0.0, 0); dok = check_desc(items, desc)
        redo = [k for k, o in enumerate(dok) if not o]
        print(f"descriptions passing first try: {np.mean(dok):.1%}; retrying {len(redo)}")
        if redo:
            d2 = write_desc([items[k] for k in redo], 0.7, 1); o2 = check_desc([items[k] for k in redo], d2)
            for k, d, o in zip(redo, d2, o2):
                if o: desc[k], dok[k] = d, True
        nh = len(HARMFUL)
        bidx = [k for k in range(len(BENIGN)) if dok[nh + k]]
        why = [""] * len(BENIGN); wok = [False] * len(BENIGN)
        def write_why(ks, temp, seed):
            return [str(t).strip() for t in chat_many(TEACHER, [WHY_P.format(prompt=items[nh + k][0][:3000], desc=desc[nh + k]) for k in ks],
                                                      temperature=temp, max_tokens=80, seed_base=seed)]
        def check_why(ks, ws):
            _, cv = judge_json([CHECK_P.format(prompt=items[nh + k][0][:3000], thought=desc[nh + k] + " " + w) for k, w in zip(ks, ws)], "ok", 40)
            return [bool(c.get("ok") is True) and assess_ok_code(w) and len(sentences(w)) <= 2 for c, w in zip(cv, ws)]
        if bidx:
            ws = write_why(bidx, 0.0, 0); os_ = check_why(bidx, ws)
            for k, w, o in zip(bidx, ws, os_): why[k], wok[k] = w, o
        redo = [k for k in bidx if not wok[k]]
        if redo:
            ws = write_why(redo, 0.7, 1); os_ = check_why(redo, ws)
            for k, w, o in zip(redo, ws, os_):
                if o: why[k], wok[k] = w, True
        stop_server()
        T2 = dict(desc_harmful=desc[:nh], desc_harmful_ok=dok[:nh], desc_benign=desc[nh:], desc_benign_ok=dok[nh:], why=why, why_ok=wok, teacher=TEACHER)
        T2 = {k: [bool(x) if isinstance(x, (bool, np.bool_)) else x for x in v] if isinstance(v, list) else v for k, v in T2.items()}
        json.dump(T2, open(f"{ARMS_DIR}/teacher_v3b.json", "w"), ensure_ascii=False)
    print(f"harmful descriptions kept {np.mean(T2['desc_harmful_ok']):.1%} | harmless descriptions kept {np.mean(T2['desc_benign_ok']):.1%}"
          f" | why-fine kept {np.mean(T2['why_ok']):.1%}")
    for (_, r), d, o in list(zip(HARMFUL, T2["desc_harmful"], T2["desc_harmful_ok"]))[:3]: print(f"  [{r['language']}] {o} {d!r}")
    for (_, r), d, w in list(zip(BENIGN, T2["desc_benign"], T2["why"]))[:2]: print(f"  [harmless] {d!r} + {w!r}")
""")

md("## 2. Build the two arms")

code(r"""
def split_reason(reason, restates, alt_from):
    # returns (why-harmful part, alternative part) using the teacher's sentence numbers; the text itself is unchanged
    ss = sentences(reason)
    if restates and len(ss) > 1: ss, alt_from = ss[1:], max(0, alt_from - 1)
    if alt_from <= 0 or alt_from > len(ss): harm, alt = ss, []
    elif alt_from == 1: harm, alt = ss[:1], ss[1:]          # first sentence is already a redirect: keep it as the reasoning
    else: harm, alt = ss[:alt_from - 1], ss[alt_from - 1:]
    return " ".join(harm), " ".join(alt)

def harmful_reply(arm, lang, harm, alt, desc=""):
    lead, o = LEAD[lang], OPENING[lang]
    body = f"{harm} {o}" if arm.startswith("reason_first") else f"{o} {harm}"
    return f"{lead} " + (f"{desc} " if desc else "") + body + (f" {alt}" if alt else "")

def drop_refusal_sentences(harm, lang):
    # v2: no refusal phrase may appear before the decision (Aya's non-English reasons often contain one)
    ss = sentences(harm); keep = [x for x in ss if not re.search(REFUSAL_IN[lang], x)]
    return " ".join(keep) if keep else ""

new_harm, drop, n_refusal_removed = {}, set(), 0
for k, ((i, r), s) in enumerate(zip(HARMFUL, T["split"])):
    harm, alt = split_reason(reason_of(r), s["restates"], s["alt_from"])
    if VERSION == 2:
        h2 = drop_refusal_sentences(harm, r["language"]); n_refusal_removed += h2 != harm
        if not h2 or not T2["desc_harmful_ok"][k]: drop.add(i); continue
        new_harm[i] = (h2, alt, T2["desc_harmful"][k])
    else: new_harm[i] = (harm, alt)
if VERSION == 2:
    new_benign = {i: f"{T2['desc_benign'][k]} {T2['why'][k]}" for k, (i, r) in enumerate(BENIGN) if T2["desc_benign_ok"][k] and T2["why_ok"][k]}
    print(f"harmful reasons with a refusal sentence removed: {n_refusal_removed} | harmful rows dropped: {len(drop)}")
else:
    new_benign = {i: t for (i, r), t, o in zip(BENIGN, T["thought"], T["thought_ok"]) if o}
drop |= {i for i, _ in BENIGN} - set(new_benign)

ARMS = {}
for arm in NEW_ARMS:
    rows = []
    for i, r in enumerate(SRC):
        if SMOKE and r["kind"] == "harmful" and i not in new_harm and i not in drop: continue
        if SMOKE and r["kind"] == "benign_lookalike" and i not in new_benign and i not in drop: continue
        if i in drop: continue
        m = [dict(x) for x in r["messages"]]
        if r["kind"] == "harmful":
            m[-1]["content"] = harmful_reply(arm, r["language"], *new_harm[i])
        elif r["kind"] == "benign_lookalike":
            m[-1]["content"] = f"{LEAD['english']} {new_benign[i]}\n\n{m[-1]['content'].lstrip()}"
        rows.append({**r, "messages": m})
    if SMOKE: rows = rows[:SMOKE_ROWS * 4]
    ARMS[arm] = rows
    with open(f"{ARMS_DIR}/train_{arm}.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")

a, b = ARMS[NEW_ARMS[0]], ARMS[NEW_ARMS[1]]
assert [x["messages"][:-1] for x in a] == [x["messages"][:-1] for x in b], "prompts or order differ between the new arms"
assert all(x["messages"][-1] == y["messages"][-1] for x, y in zip(a, b) if x["kind"] != "harmful"), "a non-harmful reply differs"
assert all(sorted(x["messages"][-1]["content"].split()) == sorted(y["messages"][-1]["content"].split()) for x, y in zip(a, b) if x["kind"] == "harmful"), \
    "harmful replies must contain the same words in both arms (only the order differs)"
print("same prompts, order and harmless rows in both arms; harmful replies differ only in where the refusal sits")
print({arm: dict(collections.Counter(x["kind"] for x in v)) for arm, v in ARMS.items()}, "| dropped look-alikes:", len(drop))
words = lambda t: len(t.split())
print(f"mean words before the refusal in harmful replies: {NEW_ARMS[0]}",
      round(np.mean([words(x["messages"][-1]["content"].split(OPENING[x["language"]])[0]) for x in a if x["kind"] == "harmful"]), 1),
      f"| {NEW_ARMS[1]}", round(np.mean([words(x["messages"][-1]["content"].split(OPENING[x["language"]])[0]) for x in b if x["kind"] == "harmful"]), 1))
for arm in NEW_ARMS:
    for kind in ["harmful", "benign_lookalike"]:
        x = next(x for x in ARMS[arm] if x["kind"] == kind)
        print(f"\n[{arm} | {kind}] {x['messages'][0]['content'][:160]!r}\n  -> {x['messages'][-1]['content'][:420]!r}")
nonen = [x for x in a if x["kind"] == "harmful" and x["language"] != "english"][:2]
for x in nonen: print(f"\n[{NEW_ARMS[0]} | {x['language']}] -> {x['messages'][-1]['content'][:300]!r}")
# the words right after the shared part (lead-in, + description in v2) should NOT be one template for harmless and another for harmful
def after_shared(t):
    t = t.split(" ", len(LEAD["english"].split()))[-1]
    if VERSION == 2: t = " ".join(sentences(t)[1:])
    return " ".join(t.split()[:2])
for arm in NEW_ARMS:
    eng = [x for x in ARMS[arm] if x["kind"] != "general" and x.get("language", "english") == "english"]
    for kind in ["harmful", "benign_lookalike"]:
        print(f"{arm:16s} {kind:17s} first words after the shared part:",
              collections.Counter(after_shared(x["messages"][-1]["content"]) for x in eng if x["kind"] == kind).most_common(5))
""")

md("## 3. Train both arms with AlignTune (with intermediate checkpoints)")

code(r"""
import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import shutil
# free the GPU before training: stop any vLLM server and drop models left from an earlier quick check in this runtime
stop_server()
for _v in ["base", "m", "pm"]:
    if _v in globals(): del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
_free, _tot = torch.cuda.mem_get_info()
print(f"GPU free before training: {_free/1e9:.1f} / {_tot/1e9:.1f} GB")
if _free / _tot < 0.9: print("WARNING: something else is still on the GPU -- restart the runtime (installed packages are kept) and run the cells again")
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
tok = AutoTokenizer.from_pretrained(STUDENT)
CHAT_TEMPLATE = open(f"{ROOT}/train/chat_template.jinja").read()      # the plain-text template every arm was trained with
tok.chat_template = CHAT_TEMPLATE
def to_prompt_completion(rows):
    return [dict(prompt=r["messages"][:-1], completion=[{"role": "assistant", "content": r["messages"][-1]["content"].lstrip()}], kind=r["kind"])
            for r in rows if r["messages"][-1]["role"] == "assistant"]
PC = {a_: to_prompt_completion([json.loads(l) for l in open(f"{ARMS_DIR}/train_{a_}.jsonl")]) for a_ in NEW_ARMS}
if SMOKE: PC = {a_: v[:SMOKE_ROWS] for a_, v in PC.items()}
def boundary_ok(x):
    pre = tok(tok.apply_chat_template(x["prompt"], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    full = tok(tok.apply_chat_template(x["prompt"] + x["completion"], tokenize=False), add_special_tokens=False).input_ids
    return full[:len(pre)] == pre
for a_ in NEW_ARMS:
    bad = [i for i, x in enumerate(PC[a_][:2000]) if not boundary_ok(x)]
    print(f"{a_:13s} {len(PC[a_])} rows | prompt not a token prefix of prompt + reply (first 2,000): {len(bad)}",
          collections.Counter(PC[a_][i]["kind"] for i in bad).most_common())

class SaveAdapterEvery(TrainerCallback):
    # saves the LoRA adapter every CKPT_EVERY optimizer steps to OUT/<arm>/ckpt-<step> (adapter files only, ~40 MB each)
    def __init__(self, arm): self.arm = arm
    def on_step_end(self, args, state, control, model=None, **kw):
        s = state.global_step
        if CKPT_EVERY and s % CKPT_EVERY == 0 and s < state.max_steps and model is not None:
            if not hasattr(model, "peft_config"): print("  (checkpoint skipped: model is not a PEFT model)"); return
            d = f"{OUT}/{self.arm}/ckpt-{s}"; model.save_pretrained(d); tok.save_pretrained(d); print(f"  saved {d}", flush=True)

from aligntune.core.backend_factory import create_sft_trainer
def train_arm(arm):
    final = f"{OUT}/{arm}/final"
    if os.path.exists(f"{final}/adapter_model.safetensors"): print(f"{arm}: already trained, skipping"); return
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
    trainer.dataset, trainer.eval_dataset = ds, None
    _cbs = trainer.get_hf_callbacks
    trainer.get_hf_callbacks = lambda: list(_cbs()) + [SaveAdapterEvery(arm)]     # AlignTune passes these to the TRL SFTTrainer
    t0 = time.time()
    try: res = trainer.train()
    except torch.OutOfMemoryError:
        if TRAIN["batch"] == 2: raise
        print("out of GPU memory at batch 4 -> retrying with batch 2 x grad_accum 16 (same effective batch of 32)")
        del trainer; gc.collect(); torch.cuda.empty_cache()
        TRAIN.update(batch=2, grad_accum=16); return train_arm(arm)
    except Exception as e:
        st = getattr(getattr(trainer, "trainer", None), "state", None)
        if st is None or st.global_step == 0: raise
        print(f"warning: AlignTune raised after training ({str(e)[:200]}); saving the trained adapter anyway")
        res = dict(steps=st.global_step, final_loss=float(pd.DataFrame(st.log_history).get("loss", pd.Series([np.nan])).dropna().iloc[-1]))
    mins = (time.time() - t0) / 60
    os.makedirs(final, exist_ok=True); trainer.trainer.save_model(final); trainer.tokenizer.save_pretrained(final)
    pd.DataFrame(trainer.trainer.state.log_history).to_csv(f"{OUT}/{arm}/log.csv", index=False)
    json.dump(dict(minutes=mins, steps=res.get("steps"), final_loss=res.get("final_loss"), rows=len(ds), lora=LORA, train=dict(TRAIN), grad_ckpt=GRAD_CKPT,
                   ckpt_every=CKPT_EVERY, smoke=SMOKE), open(f"{OUT}/{arm}/run.json", "w"), indent=1)
    print(f"{arm}: {res.get('steps')} steps in {mins:.1f} min, final loss {res.get('final_loss'):.4f}")
    del trainer; gc.collect(); torch.cuda.empty_cache()

for arm in (ONLY_ARMS or NEW_ARMS):
    print(f"\n===== {arm} ====="); train_arm(arm)
    print("  checkpoints on Drive:", sorted(d for d in os.listdir(f"{OUT}/{arm}") if d.startswith("ckpt-") or d == "final"))
""")

md(r"""
## 4. Quick check: does the model reason first, then decide?

Greedy replies to held-out prompts. Expected for both arms: harmful **and** harmless look-alike replies start with the lead-in; in `reason_first` the refusal sentence appears after the reasoning (later word position), in `lead_first` right after the lead-in; harmless prompts are answered. Not the evaluation (step 3).
""")

code(r"""
from transformers import AutoModelForCausalLM
from peft import PeftModel
EV = pd.read_json(f"{DATA_DIR}/eval_items.jsonl", lines=True)
n = 8 if SMOKE else 40
probe = pd.concat([EV[EV.set == s].sample(min(n, (EV.set == s).sum()), random_state=SEED)
                   for s in ["wgm_harmful_vanilla", "wgm_harmful_adv", "wgm_harmless_vanilla", "wgm_harmless_adv"]])
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
def generate(model, prompts, max_new=160, bs=16):
    out = []
    for i in range(0, len(prompts), bs):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(): o = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        out += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in o]
    return out
base = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map="cuda").eval()
CHECK = probe[["set", "prompt"]].copy()
for arm in NEW_ARMS:
    if not os.path.exists(f"{OUT}/{arm}/final/adapter_model.safetensors"): continue
    m = PeftModel.from_pretrained(base, f"{OUT}/{arm}/final").eval(); CHECK[arm] = generate(m, probe.prompt.tolist()); base = m.unload()
    del m; gc.collect(); torch.cuda.empty_cache()
CHECK.to_csv(f"{OUT}/quick_check_v3{'' if VERSION == 1 else 'b'}.csv", index=False)
cols = [c for c in NEW_ARMS if c in CHECK]
def refusal_word_pos(t):
    k = t.find(OPENING["english"]); return len(t[:k].split()) if k >= 0 else np.nan
summ = pd.DataFrame({c: CHECK.assign(lead=CHECK[c].str.startswith(LEAD["english"]), refuses=CHECK[c].str.contains(OPENING["english"], regex=False),
                                     pos=CHECK[c].map(refusal_word_pos)).groupby("set")[["lead", "refuses", "pos"]].mean().round(2).stack()
                     for c in cols})
print("share starting with the lead-in / share containing the refusal / mean word position of the refusal:"); print(summ)
print("\nfirst words after the shared part, harmful prompts (if one template = comply and another = refuse, the decision is still made right there):")
for c in cols:
    h_ = CHECK[CHECK.set.str.startswith("wgm_harmful")]
    for ref in [True, False]:
        sub = h_[h_[c].str.contains(OPENING["english"], regex=False) == ref][c]
        print(f"  {c:16s} {'refused ' if ref else 'complied'}", collections.Counter(after_shared(t) for t in sub).most_common(5))
for s in ["wgm_harmful_adv", "wgm_harmless_adv"]:
    r = CHECK[CHECK.set == s].iloc[0]
    print(f"\n[{s}] {r.prompt[:200]!r}")
    for c in cols: print(f"  {c:13s}: {r[c][:300]!r}")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("written", sys.argv[1], len(cells), "cells")
