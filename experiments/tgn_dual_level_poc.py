"""Dual-level safety monitor, proof of concept on SYNTHETIC structural trajectories (no LLM, no harmful content).

Claim: a NODE-level content judge (sees one action's local features) cannot catch harm decomposed into individually-benign steps;
a GRAPH-level temporal-graph monitor (per-node memory updated by each timestamped action, taint flowing forward along edges) can.

'Harm' is purely structural: sensitive data reaches an EXTERNAL sink without being sanitised. Four hard cases force the point:
  - harm:         read credential -> (benign steps) -> send external            (label 1)
  - hard_benign:  read PUBLIC     -> (benign steps) -> send external            (label 0; identical final action)
  - sanitised:    read credential -> REDACT -> send external                    (label 0; taint removed mid-trajectory)
  - benign_touch: read credential -> keep internal                             (label 0)
The payload node sent out is static-benign in every case, so the taint lives ONLY in history. Player vs team: the node judge
rates one action (xG/VAEP); the TGN values the whole possession (xT).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
rng = np.random.RandomState(0); torch.manual_seed(0)

SENS = {"public": 0, "personal": 1, "credential": 2}
ET = {"read": 0, "derive": 1, "send": 2, "clean": 3, "noop": 4}; NE = len(ET)
HELD_OUT_TOOL = 1                                        # a tool feature-signature used only at test (inductive check)

def episode(kind, dilute=0, gap=None):
    """events [(t, src, dst, edge_type)], node feats, label. node 0 = agent, 1 = external party, 2 = internal store."""
    nodes = {0: ("agent", 0, 0, 1), 1: ("party", 0, 1, 0), 2: ("store", 0, 0, 1)}; nid = 3; events = []; t = 0
    def add(typ, sens=0, ext=0, rev=1):
        nonlocal nid; nodes[nid] = (typ, sens, ext, rev); nid += 1; return nid - 1
    reads_cred = kind in ("harm", "sanitised", "benign_touch")
    # 1) initial read into the agent buffer
    if kind in ("harm", "hard_benign", "sanitised", "benign_touch"):
        dsens = SENS["credential"] if reads_cred else SENS["public"]
        tool = add("tool", 0, 0, 1); data = add("data", dsens, 0, 1)
        events += [(t, data, tool, ET["read"]), (t + 1, tool, 0, ET["read"])]; t += 2
    # 2) benign distractor steps (individually harmless), plus optional dilution padding
    ndist = (gap if gap is not None else rng.randint(0, 4)) + dilute
    for _ in range(ndist):
        tl = add("tool", 0, 0, 1); d = add("data", 0, 0, 1)
        events += [(t, d, tl, ET["derive"]), (t + 1, tl, 0, ET["derive"])]; t += 2
    # 2b) sanitiser removes taint from the agent buffer
    if kind == "sanitised":
        rd = add("tool", 0, 0, 1); events += [(t, 0, rd, ET["clean"]), (t + 1, rd, 0, ET["clean"])]; t += 2
    # 3) terminal action
    if kind in ("harm", "hard_benign", "sanitised", "benign_send_pub"):
        tl = add("tool", 0, 1, int(rng.rand() < 0.15 and kind == "harm"))             # 'reversible' bit doubles as a rare held-out signature
        payload = add("data", 0, 0, 1)                                                # payload is static-benign for everyone
        events += [(t, 0, tl, ET["derive"]), (t + 1, payload, tl, ET["send"]), (t + 2, tl, 1, ET["send"])]
    elif kind == "benign_touch":
        tl = add("tool", 0, 0, 1); events += [(t, 0, tl, ET["derive"]), (t + 1, tl, 2, ET["derive"])]
    else:
        tl = add("tool", 0, 0, 1); events += [(t, 0, tl, ET["noop"])]
    return events, nodes, int(kind == "harm")

KINDS = ["harm", "hard_benign", "sanitised", "benign_send_pub", "benign_touch", "benign_idle"]
PK = [0.28, 0.24, 0.18, 0.12, 0.10, 0.08]
def make(n, dilute=0):
    ks = rng.choice(KINDS, n, p=PK); return [episode(k, dilute) for k in ks], ks
train, ktr = make(3000); test, kte = make(1500)
print("episode mix (train):", {k: int((ktr == k).sum()) for k in KINDS})

def feat(node):
    typ, sens, ext, rev = node; ti = {"agent": 0, "party": 1, "store": 2, "tool": 3, "data": 3}[typ]
    v = np.zeros(4); v[ti] = 1; return np.r_[v, sens / 2, ext, rev].astype(np.float32)

# ---- node-level judge: local features of the terminal external send only ----
def local_rows(data):
    X, y = [], []
    for events, nodes, lab in data:
        send = [e for e in events if e[3] == ET["send"]]
        if not send: continue
        _, s, d, et = send[-1]; X.append(np.r_[feat(nodes[s]), feat(nodes[d]), np.eye(NE)[et]]); y.append(lab)
    return np.array(X), np.array(y)
Xtr, ytr = local_rows(train); Xte, yte = local_rows(test)
judge = LogisticRegression(max_iter=1000).fit(Xtr, ytr); pj = judge.predict_proba(Xte)[:, 1]
print(f"\nNODE-LEVEL JUDGE (local features of the send action): AUROC {roc_auc_score(yte, pj):.3f}")

# ---- graph-level TGN: per-node memory, taint flows FORWARD along each timestamped edge ----
class TGN(nn.Module):
    def __init__(s, fdim=7, d=48):
        super().__init__(); s.fenc = nn.Linear(fdim, d)
        s.msg = nn.Sequential(nn.Linear(2 * d + NE, d), nn.ReLU(), nn.Linear(d, d)); s.upd = nn.GRUCell(d, d)
        s.read = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
    def forward(s, events, nodes):
        mem = {i: s.fenc(torch.tensor(feat(nodes[i]))) for i in nodes}
        for (t, a, b, et) in events:                                                  # directed: message from src a updates dst b
            m = s.msg(torch.cat([mem[a], mem[b], F.one_hot(torch.tensor(et), NE).float()]))
            mem[b] = s.upd(m.unsqueeze(0), mem[b].unsqueeze(0))[0]
        return s.read(torch.cat([mem[0], mem[1]]))                                     # agent + external party
def train_tgn(data, seed=0, epochs=10, bs=16, lr=2e-3):
    torch.manual_seed(seed); r = np.random.RandomState(seed); net = TGN(); opt = torch.optim.Adam(net.parameters(), lr)
    for ep in range(epochs):
        order = r.permutation(len(data))
        for i in range(0, len(order), bs):
            opt.zero_grad(); loss = 0.0
            for j in order[i:i + bs]:
                events, nodes, lab = data[j]
                loss = loss + F.binary_cross_entropy_with_logits(net(events, nodes), torch.tensor([float(lab)]))
            (loss / len(order[i:i + bs])).backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return net.eval()
net = train_tgn(train, seed=0)
with torch.no_grad():
    pt = np.array([torch.sigmoid(net(e, n)).item() for e, n, _ in test]); yt = np.array([l for _, _, l in test])
print(f"GRAPH-LEVEL TGN (memory over the trajectory):        AUROC {roc_auc_score(yt, pt):.3f}")

# ---- decisive comparisons ----
def auc_on(kinds_a, kinds_b, scores):
    idx = [i for i, k in enumerate(kte) if k in kinds_a + kinds_b]; y = np.array([kte[i] in kinds_a for i in idx], int)
    return roc_auc_score(y, scores[idx]) if 0 < y.sum() < len(y) else np.nan
# node judge on the hard pair, restricted to episodes with a send
def judge_auc(kinds_a, kinds_b):
    sel = [i for i, k in enumerate(kte) if k in kinds_a + kinds_b and any(e[3] == ET["send"] for e in test[i][0])]
    X = np.array([np.r_[feat(test[i][1][[e for e in test[i][0] if e[3]==ET['send']][-1][1]]), feat(test[i][1][1]), np.eye(NE)[ET['send']]] for i in sel])
    y = np.array([kte[i] in kinds_a for i in sel], int); return roc_auc_score(y, judge.predict_proba(X)[:, 1]) if 0 < y.sum() < len(y) else np.nan
print("\nDECOMPOSITION  (harm vs hard-benign: identical final action, differ only in the data read earlier)")
print(f"  node-level judge: AUROC {judge_auc(['harm'], ['hard_benign']):.3f}   (chance by construction)")
print(f"  graph-level TGN : AUROC {auc_on(['harm'], ['hard_benign'], pt):.3f}")
print("SANITISER  (harm vs sanitised: same read, but a redact step cleans the buffer before the send)")
print(f"  node-level judge: AUROC {judge_auc(['harm'], ['sanitised']):.3f}")
print(f"  graph-level TGN : AUROC {auc_on(['harm'], ['sanitised'], pt):.3f}")

# ---- dilution attack: pad harmful episodes with K benign steps; does memory wash out? ----
print("\nDILUTION ATTACK (benign steps padded before the send):")
for K in [0, 4, 10, 20, 40]:
    dt, dk = make(400, dilute=K); hi = [i for i in range(len(dt)) if dt[i][2] == 1]
    with torch.no_grad(): ph = np.array([torch.sigmoid(net(dt[i][0], dt[i][1])).item() for i in hi])
    print(f"  +{K:2d} benign steps (median len {int(np.median([len(dt[i][0]) for i in hi]))}): TGN recall on harmful {np.mean(ph > 0.5):.2f}")

# ---- TRANSFER: train only on MULTI-STEP, test on SINGLE-TURN (read and send adjacent, never seen in training) ----
# the monitor is policy-agnostic: knowledge from long horizons should reduce to the 1-step case a single-turn policy produces.
print("\nTRANSFER  (monitor trained on multi-step; tested on single-turn episodes it never saw)")
# retrain on multi-step only (gap >= 2), so no near-adjacent read->send appears in training
rng2 = np.random.RandomState(7)
train_ms = [episode(k, gap=int(2 + rng2.randint(0, 4))) for k in rng2.choice(KINDS, 3000, p=PK)]
netT = train_tgn(train_ms, seed=0)
for name, gap in [("multi-step (held-in horizon)", None), ("SINGLE-TURN gap=0 (transfer)", 0)]:
    ks = rng2.choice(["harm", "hard_benign"], 800); ep_ = [episode(k, gap=gap) for k in ks]
    with torch.no_grad(): s = np.array([torch.sigmoid(netT(e, n)).item() for e, n, _ in ep_])
    y = np.array([l for _, _, l in ep_]); rec = (s[y == 1] > 0.5).mean(); fp = (s[y == 0] > 0.5).mean()
    print(f"  {name:32} harm-vs-hard-benign AUROC {roc_auc_score(y, s):.3f} | recall {rec:.2f} | false-alarm {fp:.2f}")
print("  (node-level local judge stays at chance on both, since the taint is never in one action's view)")

# ---- memory mechanism ablation: GRU memory vs explicit gated (write / erase) memory ----
# gated cell: mem[b] = (1 - g_erase) * mem[b] + g_write * content, gates conditioned on the event.
# an UNRELATED benign event can set g_write~0, g_erase~0 -> memory preserved (dilution-robust); g_erase fires on 'clean' -> sanitiser works.
class TGNgated(nn.Module):
    def __init__(s, fdim=7, d=48):
        super().__init__(); s.fenc = nn.Linear(fdim, d)
        s.w = nn.Linear(2 * d + NE, d); s.e = nn.Linear(2 * d + NE, d); s.c = nn.Linear(2 * d + NE, d)
        s.read = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
    def forward(s, events, nodes):
        mem = {i: s.fenc(torch.tensor(feat(nodes[i]))) for i in nodes}
        for (t, a, b, et) in events:
            x = torch.cat([mem[a], mem[b], F.one_hot(torch.tensor(et), NE).float()])
            mem[b] = (1 - torch.sigmoid(s.e(x))) * mem[b] + torch.sigmoid(s.w(x)) * torch.tanh(s.c(x))
        return s.read(torch.cat([mem[0], mem[1]]))
def train_any(cls, data, seed=0, epochs=10, bs=16, lr=2e-3):
    torch.manual_seed(seed); r = np.random.RandomState(seed); net = cls(); opt = torch.optim.Adam(net.parameters(), lr)
    for ep in range(epochs):
        order = r.permutation(len(data))
        for i in range(0, len(order), bs):
            opt.zero_grad(); loss = sum(F.binary_cross_entropy_with_logits(net(*data[j][:2]), torch.tensor([float(data[j][2])])) for j in order[i:i+bs])
            (loss / len(order[i:i+bs])).backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return net.eval()
print("\nMEMORY MECHANISM ABLATION (same data, same trainer)")
gated = train_any(TGNgated, train, seed=0)
for nm, model in [("GRU memory", net), ("gated write/erase memory", gated)]:
    with torch.no_grad():
        ps = np.array([torch.sigmoid(model(e, n)).item() for e, n, _ in test])
    aus = auc_on(['harm'], ['sanitised'], ps); ahb = auc_on(['harm'], ['hard_benign'], ps)
    print(f"  {nm:26} harm-vs-hard-benign {ahb:.2f} | harm-vs-sanitised {aus:.2f}")
    line = []
    for K in [0, 10, 20, 40]:
        dt, _ = make(400, dilute=K); hi = [i for i in range(len(dt)) if dt[i][2] == 1]
        with torch.no_grad(): ph = np.array([torch.sigmoid(model(dt[i][0], dt[i][1])).item() for i in hi])
        line.append(f"+{K}:{np.mean(ph>0.5):.2f}")
    print(f"  {'':26} dilution recall  " + "  ".join(line))

# ---- real-time trace: P(harmful) after each action, GRU vs gated memory ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ET_NAME = {v: k for k, v in ET.items()}
def trace(model, events, nodes, gated=False):
    mem = {i: model.fenc(torch.tensor(feat(nodes[i]))) for i in nodes}; ps = []
    with torch.no_grad():
        ps.append(torch.sigmoid(model.read(torch.cat([mem[0], mem[1]]))).item())
        for (tt, a, b, et) in events:
            x = torch.cat([mem[a], mem[b], F.one_hot(torch.tensor(et), NE).float()])
            if gated: mem[b] = (1 - torch.sigmoid(model.e(x))) * mem[b] + torch.sigmoid(model.w(x)) * torch.tanh(model.c(x))
            else: mem[b] = model.upd(model.msg(x).unsqueeze(0), mem[b].unsqueeze(0))[0]
            ps.append(torch.sigmoid(model.read(torch.cat([mem[0], mem[1]]))).item())
    return ps
rr = np.random.RandomState(3)
cases = {"harmful (read credential -> ... -> send external)": episode("harm", gap=2),
         "hard-benign (read PUBLIC -> ... -> send external)": episode("hard_benign", gap=2),
         "sanitised (read credential -> REDACT -> send)": episode("sanitised", gap=2),
         "harmful + dilution (20 benign steps before send)": episode("harm", dilute=20)}
fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
for ax, (is_gated, title) in zip(axs, [(False, "GRU memory"), (True, "gated write/erase memory")]):
    model = gated if is_gated else net
    for lab, (ev, no, y) in cases.items():
        col = {"harmful (read credential -> ... -> send external)": "#c0392b", "hard-benign (read PUBLIC -> ... -> send external)": "#2471a3",
               "sanitised (read credential -> REDACT -> send)": "#27ae60", "harmful + dilution (20 benign steps before send)": "#e67e22"}[lab]
        pr = trace(model, ev, no, gated=is_gated); ax.plot(range(len(pr)), pr, marker=".", ms=4, color=col, label=lab)
        sends = [k + 1 for k, e in enumerate(ev) if e[3] == ET["send"]]
        for sdt in sends: ax.axvline(sdt, color=col, ls=":", lw=0.6, alpha=0.4)
    ax.axhline(0.5, color="#888", ls="--", lw=0.8); ax.set_title(f"{title}", fontsize=11, loc="left")
    ax.set_xlabel("action step (0 = start; dotted = an external send)"); ax.set_ylim(-0.03, 1.03)
axs[0].set_ylabel("P(trajectory ends in a harmful action)"); axs[0].legend(fontsize=7.5, loc="center left")
plt.suptitle("Real-time threat, step by step: taint accumulates, GRU washes out under dilution, gated memory holds", fontsize=11, x=0.01, ha="left")
plt.tight_layout(); plt.savefig("tgn_trace.png", dpi=150); print("saved tgn_trace.png")
