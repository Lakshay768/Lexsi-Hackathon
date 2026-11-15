# Core training code for views_jepa.ipynb (stage 1: next-token or LLM-JEPA on request <-> tool-call pairs; stage 2: chat refusal SFT).
# Inserted verbatim into the notebook; also imported by the local CPU test. Expects globals: tok, STUDENT, LORA, S1, S2, JEPA, SEED.
import math, random, time, json, os, gc
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

def render(prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

def render_pair(prompt, reply):
    p = render(prompt)
    full = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}], tokenize=False)
    assert full.startswith(p), "chat template: prompt is not a prefix of prompt+reply"
    return p, full

def tokenize_sft(prompt, reply, max_len):
    # token ids of prompt+reply; labels only on reply tokens (completion-only loss, as in the AlignTune/TRL runs)
    p, full = render_pair(prompt, reply)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_len)
    labels = [i if s >= len(p) else -100 for i, (s, e) in zip(enc["input_ids"], enc["offset_mapping"])]
    return dict(input_ids=enc["input_ids"], labels=labels)

def pad_batch(items, dev):
    n = max(len(x["input_ids"]) for x in items)
    pid = tok.pad_token_id
    ids = torch.tensor([x["input_ids"] + [pid] * (n - len(x["input_ids"])) for x in items], device=dev)
    lab = torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in items], device=dev)
    att = torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in items], device=dev)
    return ids, att, lab

def inner(model):
    # the transformer without the LM head (LoRA layers are injected in place, so they are active here)
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model

def last_states(model, texts, layer, dev, max_len=512):
    # hidden state of the final real token of each text (right padding); layer -1 = last layer (after the final norm)
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=max_len).to(dev)
    out = inner(model)(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=(layer != -1), use_cache=False)
    h = out.last_hidden_state if layer == -1 else out.hidden_states[layer]
    idx = enc.attention_mask.sum(1) - 1
    return h[torch.arange(len(idx), device=dev), idx].float()

def multi_states(model, texts, layers, dev, max_len=512):
    # {layer: [n, d]} at the final real token, one forward pass for all layers (-1 = last layer after the final norm)
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=max_len).to(dev)
    out = inner(model)(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True, use_cache=False)
    idx = enc.attention_mask.sum(1) - 1; ar = torch.arange(len(idx), device=dev)
    return {l: (out.last_hidden_state if l == -1 else out.hidden_states[l])[ar, idx].float() for l in layers}

def info_nce(a, b, tau):
    a = F.normalize(a - a.mean(0, keepdim=True), dim=-1); b = F.normalize(b - b.mean(0, keepdim=True), dim=-1)
    logits = a @ b.T / tau; lab_ = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, lab_) + F.cross_entropy(logits.T, lab_)), float((logits.argmax(1) == lab_).float().mean())

def sup_con(a, b, labels, tau):
    # supervised contrastive across views: a_i's positives are every b_j with the same label (b_j is a DIFFERENT request); symmetric
    a = F.normalize(a - a.mean(0, keepdim=True), dim=-1); b = F.normalize(b - b.mean(0, keepdim=True), dim=-1)
    lab = torch.tensor([hash(x) % (2 ** 31) for x in labels], device=a.device); pos = (lab[:, None] == lab[None, :])
    def one(lg):
        return (torch.logsumexp(lg, 1) - torch.logsumexp(lg.masked_fill(~pos, float("-inf")), 1)).mean()
    lg = a @ b.T / tau
    acc = float(pos[torch.arange(len(a)), lg.argmax(1)].float().mean())
    return 0.5 * (one(lg) + one(lg.T)), acc

def corrupt(text, p, seed):
    # bad view: the same request with a share of its words dropped (a corruption of one observation)
    r = random.Random(seed); w = text.split()
    keep = [x for x in w if r.random() >= p] or w[:1]
    return " ".join(keep)

def turn(role, text):
    # one chat turn on its own, ending in the end-of-turn token (as the reference LLM-JEPA code reads both views)
    t = tok.apply_chat_template([{"role": role, "content": text}], tokenize=False).rstrip()
    return t if t.endswith(tok.eos_token) else t + tok.eos_token

def views(ex, kind, i):
    if kind == "good_f":   # faithful to the reference code: both views chat-formatted, read at the end-of-turn token, no [PRED]
        return turn("user", ex["query"]), turn("assistant", ex["call"])
    # (predictor-side text, target-side text) for one stage-1 example
    pred_side = render(ex["query"]) + JEPA["pred_str"] * JEPA["k_pred"]
    if kind in ("good", "good_c", "good_m"): tgt = ex["call"]                       # the same request as an action: the tool call
    elif kind in ("bad", "bad_c", "bad_m"): tgt = corrupt(ex["query"], JEPA["drop_p"], SEED * 100003 + i)
    else: raise ValueError(kind)
    return pred_side, BOS + tgt

def new_lora_model(dev):
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)   # identical LoRA initialisation in every arm
    base = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to(dev)
    base.config.use_cache = False
    cfg = LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"], target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    return model

def load_lora_model(path, dev):
    base = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to(dev)
    base.config.use_cache = False
    return PeftModel.from_pretrained(base, path, is_trainable=True)

def train_stage(model, data, cfg, dev, kind=None, lam=0.0, log_path=None, max_steps=None, order=None, xview=None, ckpt_dir=None, ckpt_every=0):
    """One epoch over `data`. data rows: dict(prompt, reply) for SFT, plus (query, call) for stage-1 JEPA views.
    loss = next-token loss on the reply + lam * (1 - cos(Pred(Enc(words view)), Enc(other view)))"""
    if cfg.get("grad_ckpt", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    torch.manual_seed(SEED)   # same dropout stream at the start of each stage in every arm
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    mb, acc = cfg["micro_batch"], cfg["grad_accum"]
    if order is None:
        order = list(range(len(data))); random.Random(SEED).shuffle(order)
    else:
        order = list(order)   # a pre-built order (intent arms: the first JEPA["batch"] items of each step are distinct intents)
    steps = math.ceil(len(order) / (mb * acc))
    if max_steps: steps = min(steps, max_steps)
    sch = get_linear_schedule_with_warmup(opt, max(1, math.ceil(cfg["warmup"] * steps)), steps)
    toks = {}
    log, t0, step = [], time.time(), 0
    use_amp = dev.startswith("cuda")
    for step in range(steps):
        idxs = order[step * mb * acc:(step + 1) * mb * acc]
        tot = dict(loss=0.0, ntp=0.0, jepa=0.0, jacc=0.0)
        n_mb = math.ceil(len(idxs) / mb)
        for k in range(n_mb):
            chunk = idxs[k * mb:(k + 1) * mb]
            items = []
            for i in chunk:
                if i not in toks: toks[i] = tokenize_sft(data[i]["prompt"], data[i]["reply"], cfg["max_seq"])
                items.append(toks[i])
            ids, att, lab = pad_batch(items, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                ntp = model(input_ids=ids, attention_mask=att, labels=lab, use_cache=False).loss
                loss = ntp
                jl = torch.zeros((), device=dev)
                if kind in ("good", "bad", "good_f") and lam > 0:
                    pv = [views(data[i], kind, i) for i in chunk]
                    pred = last_states(model, [a for a, _ in pv], JEPA["layer"], dev, cfg["max_view"])
                    tgt = last_states(model, [b for _, b in pv], JEPA["layer"], dev, cfg["max_view"])
                    jl = (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
                    loss = loss + lam * jl
            (loss / n_mb).backward()
            tot["loss"] += loss.detach().item() / n_mb; tot["ntp"] += ntp.detach().item() / n_mb; tot["jepa"] += float(jl.detach()) / n_mb
        if kind in ("good_c", "bad_c", "good_m", "bad_m") and lam > 0:
            # contrastive JEPA (anti-collapse): each request's prediction must pick out ITS OWN target among the
            # other targets in a batch of JEPA["batch"] pairs (symmetric InfoNCE on batch-centred, normalised embeddings).
            # Plain 1 - cos can be satisfied by pushing every embedding the same way, which is what happened in views_good.
            jb = idxs[:JEPA["batch"]]
            pv = [views(data[i], kind, i) for i in jb]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = last_states(model, [a for a, _ in pv], JEPA["layer"], dev, cfg["max_view"])
                tgt = last_states(model, [b for _, b in pv], JEPA["layer"], dev, cfg["max_view"])
            pred = F.normalize(pred - pred.mean(0, keepdim=True), dim=-1); tgt = F.normalize(tgt - tgt.mean(0, keepdim=True), dim=-1)
            logits = pred @ tgt.T / JEPA["tau"]; lab_ = torch.arange(len(jb), device=dev)
            jl = 0.5 * (F.cross_entropy(logits, lab_) + F.cross_entropy(logits.T, lab_))
            (lam * jl).backward()
            tot["jepa"] = float(jl.detach()); tot["jacc"] = float((logits.argmax(1) == lab_).float().mean())
            tot["loss"] += lam * tot["jepa"]
        if xview is not None and lam > 0:
            # cross-view alignment at the decision token: view 1 = a request as a chat prompt, view 2 = a request with the SAME
            # intent as an action-format prompt (good: a different, independently written command; bad: the same text wrapped).
            # Symmetric InfoNCE over JEPA["batch"] intents per step, averaged over xview["layers"].
            vb = xview["batch"](step); v1, v2 = vb[0], vb[1]; xlab = vb[2] if len(vb) > 2 else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                s1 = multi_states(model, v1, xview["layers"], dev, cfg["max_view"]); s2 = multi_states(model, v2, xview["layers"], dev, cfg["max_view"])
            parts = [info_nce(s1[l], s2[l], JEPA["tau"]) if xlab is None else sup_con(s1[l], s2[l], xlab, JEPA["tau"]) for l in xview["layers"]]
            jl = sum(x for x, _ in parts) / len(parts)
            (lam * jl).backward()
            tot["jepa"] = float(jl.detach()); tot["jacc"] = float(np.mean([a for _, a in parts])); tot["loss"] += lam * tot["jepa"]
        torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        log.append(dict(step=step + 1, lr=sch.get_last_lr()[0], **tot))
        if ckpt_dir and ckpt_every and (step + 1) % ckpt_every == 0 and step + 1 < steps:
            model.save_pretrained(f"{ckpt_dir}/step{step + 1}")   # intermediate adapter, for tracking the mechanism over training
        if (step + 1) % cfg.get("log_every", 10) == 0 or step == 0 or step + 1 == steps:
            el = time.time() - t0
            print(f"  step {step+1}/{steps}  loss {tot['loss']:.4f}  ntp {tot['ntp']:.4f}  jepa {tot['jepa']:.4f}"
                  + (f"  in-batch acc {tot['jacc']:.2f}" if (kind in ("good_c", "bad_c", "good_m", "bad_m") or xview is not None) else "") + "  "
                  f"{el/60:.1f} min, ~{el/(step+1)*(steps-step-1)/60:.1f} min left", flush=True)
    if log_path: json.dump(log, open(log_path, "w"))
    model.eval()
    return log

@torch.no_grad()
def pair_retrieval(model, queries, targets, layer, dev, with_pred=False, bs=16, faithful=False):
    """Centered-cosine retrieval: for each request (words view), rank its own tool call (action view) among all held-out calls.
    Returns top-1, top-5 and mean cosine of matched pairs. with_pred=True encodes the request through the [PRED] tokens."""
    model.eval()
    X, Y = [], []
    for i in range(0, len(queries), bs):
        if faithful:
            q, t = [turn("user", x) for x in queries[i:i + bs]], [turn("assistant", y) for y in targets[i:i + bs]]
        else:
            q = [render(x) + (JEPA["pred_str"] * JEPA["k_pred"] if with_pred else "") for x in queries[i:i + bs]]
            t = [BOS + y for y in targets[i:i + bs]]
        X.append(last_states(model, q, layer, dev).cpu()); Y.append(last_states(model, t, layer, dev).cpu())
    X, Y = torch.cat(X), torch.cat(Y)
    raw_cos = F.cosine_similarity(X, Y, dim=-1).mean().item()
    X = F.normalize(X - X.mean(0), dim=-1); Y = F.normalize(Y - Y.mean(0), dim=-1)
    S = X @ Y.T
    rank = (S > S.diag()[:, None]).sum(1)
    return dict(top1=float((rank == 0).float().mean()), top5=float((rank < 5).float().mean()), matched_cos=raw_cos,
                centered_matched_cos=float(S.diag().mean()), n=len(queries))


def intent_order(rows, step_size, n_distinct, seed):
    """Order for the intent arms: each step of `step_size` items starts with `n_distinct` items of DIFFERENT intents taken
    from as few scenarios as possible (hard negatives: alarm_set vs alarm_remove vs alarm_query), then fills the step."""
    rng = random.Random(seed)
    pools = {}
    for i, r in enumerate(rows): pools.setdefault(r["intent"], []).append(i)
    for v in pools.values(): rng.shuffle(v)
    scen = {}
    for it in pools: scen.setdefault(rows[pools[it][0]]["scenario"], []).append(it)
    order = []
    while True:
        avail = {s_: [it for it in its if pools[it]] for s_, its in scen.items()}
        avail = {s_: v for s_, v in avail.items() if v}
        if sum(len(v) for v in avail.values()) < n_distinct: break
        ss = sorted(avail, key=lambda s_: (-len(avail[s_]), rng.random()))   # big scenarios first: more same-scenario negatives
        chosen = []
        for s_ in ss:
            its = avail[s_][:]; rng.shuffle(its)
            for it in its:
                if len(chosen) < n_distinct: chosen.append(it)
            if len(chosen) >= n_distinct: break
        step = [pools[it].pop() for it in chosen]
        rest = [it for it in chosen if pools[it]] or [it for it in pools if pools[it]]
        while len(step) < step_size and rest:
            it = rest[len(step) % len(rest)]
            if pools[it]: step.append(pools[it].pop())
            rest = [x for x in rest if pools[x]]
        if len(step) < step_size: break
        order += step
    return order

@torch.no_grad()
def intent_match(model, held, intents, scen_of, layer, dev, with_pred=True, bs=16):
    """Held-out commands -> nearest intent frame (centered cosine). top1 = among all intents; within = among the intents of
    the command's own scenario (the hard part: alarm_set vs alarm_remove)."""
    model.eval()
    calls = [json.dumps({"name": it}) for it in intents]
    Y = torch.cat([last_states(model, [BOS + c for c in calls[i:i + bs]], layer, dev).cpu() for i in range(0, len(calls), bs)])
    X = torch.cat([last_states(model, [render(r["query"]) + (JEPA["pred_str"] * JEPA["k_pred"] if with_pred else "") for r in held[i:i + bs]], layer, dev).cpu()
                   for i in range(0, len(held), bs)])
    X = F.normalize(X - X.mean(0), dim=-1); Y = F.normalize(Y - Y.mean(0), dim=-1)
    S = X @ Y.T
    gold = torch.tensor([intents.index(r["intent"]) for r in held])
    top1 = float((S.argmax(1) == gold).float().mean())
    within = []
    for k, r in enumerate(held):
        cand = [j for j, it in enumerate(intents) if scen_of[it] == r["scenario"]]
        within.append(cand[int(S[k, cand].argmax())] == int(gold[k]))
    return dict(top1=top1, within_scenario=float(np.mean(within)), chance_all=1 / len(intents),
                chance_within=float(np.mean([1 / sum(scen_of[it] == r["scenario"] for it in intents) for r in held])), n=len(held))


@torch.no_grad()
def xview_match(model, pairs, layers, dev, bs=16):
    """Held-out cross-view matching: chat prompt of command u1 -> action prompt of a DIFFERENT command u2 with the same intent,
    among one pair per intent (top-1, chance 1/n), plus a channel probe (can a linear probe still tell chat from action?)."""
    model.eval()
    v1 = [a for a, _ in pairs]; v2 = [b for _, b in pairs]
    S1 = {l: [] for l in layers}; S2 = {l: [] for l in layers}
    for i in range(0, len(pairs), bs):
        a = multi_states(model, v1[i:i + bs], layers, dev); b = multi_states(model, v2[i:i + bs], layers, dev)
        for l in layers: S1[l].append(a[l].cpu()); S2[l].append(b[l].cpu())
    out = {}
    for l in layers:
        X, Y = torch.cat(S1[l]), torch.cat(S2[l])
        Xn = F.normalize(X - X.mean(0), dim=-1); Yn = F.normalize(Y - Y.mean(0), dim=-1); S = Xn @ Yn.T
        top1 = float(((S > S.diag()[:, None]).sum(1) == 0).float().mean())
        # channel probe: train on half the pairs, test on the other half
        n = len(X) // 2
        Z = torch.cat([X, Y]).numpy(); yc = np.r_[np.zeros(len(X)), np.ones(len(Y))]
        tr = np.r_[np.arange(n), len(X) + np.arange(n)]; te = np.r_[np.arange(n, len(X)), len(X) + np.arange(n, len(Y))]
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=0.05, max_iter=2000).fit((Z[tr] - mu) / sd, yc[tr])
        out[str(l)] = dict(top1=top1, chance=1 / len(X), channel_probe_acc=float(clf.score((Z[te] - mu) / sd, yc[te])))
    return out
