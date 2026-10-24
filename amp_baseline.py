#!/usr/bin/env python3
"""
Antimicrobial peptide (AMP) design pipeline: LSTM generator -> filter chain -> MIC ranker -> top-100.

Colab usage
-----------
    !python amp_baseline.py --demo                      # smoke test on synthetic data (NOT real AMPs)
    !python amp_baseline.py --data my_amps.csv          # real run

If you paste the code into a notebook cell instead, replace the last lines with, e.g.:
    main(["--demo"])
    main(["--data", "my_amps.csv", "--no_cys"])

Input CSV (--data)
------------------
    sequence   required. Peptide sequence, one-letter codes.
    mic_um     optional. MIC in micromolar (convert from ug/mL yourself:
               uM = ug/mL * 1000 / molecular_weight_Da). Column may also be called "mic".
               If present, a regressor ranks candidates by predicted potency; if not,
               a crude heuristic is used instead.
    organism   optional (also: bacterium, bacteria, species, strain, target). One row per
               (peptide, organism) measurement. Needed for --organism, which trains the potency
               ranker on one organism only, e.g.  --organism "e. coli,escherichia coli"
               (see --list_organisms for the labels in your file).

Download training peptides from the public databases yourself (DBAASP, APD, DRAMP, GRAMPA...)
and put them in that CSV format. Check each database's terms of use before sharing any data.

Outputs (in --out, default ./amp_out)
-------------------------------------
    library.csv / library.fasta   the 50,000-sequence library
    top100.csv                    ranked top-100 (cluster-capped, diversity-filtered, no near-copies of known peptides) with columns:
                                  score (= -log10 predicted MIC in uM), pred_mic_um, length, net_charge,
                                  cluster, nearest_known, nearest_identity, kmer_overlap_known
    report.txt                    gate-by-gate survival counts and novelty/diversity checks
    lstm_best.pt                  best generator weights (by validation loss)

The sequence rules enforced here (20 standard amino acids, 8-50 residues, unique, linear peptides)
are simple defaults: change them with --min_len / --max_len or edit is_valid() for your use case.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# PyTorch deterministic mode needs this before any CUDA matrix op runs; setting it at import time
# (before the notebook touches the GPU) is the safe place. Harmless otherwise.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, BOS, EOS = 0, 1, 2
STOI = {a: i + 3 for i, a in enumerate(AA)}
ITOS = {i: a for a, i in STOI.items()}
VOCAB = len(AA) + 3

# Kyte-Doolittle hydropathy
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        if deterministic:
            # needed by PyTorch's deterministic mode for some CUDA matrix ops (must be set before they run)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as e:  # old PyTorch or unsupported
                log(f"note: could not enable PyTorch deterministic mode ({e})")
    except ImportError:
        pass


# --------------------------------------------------------------------------- data
def is_valid(s, min_len, max_len):
    return min_len <= len(s) <= max_len and set(s) <= set(AA)


def geo_mean(x):
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


ORG_COLS = ("organism", "bacterium", "bacteria", "species", "strain", "target")


def load_data(path, min_len, max_len, organism_col=None):
    """Returns (df, rows): df = one row per unique valid sequence (MIC = geometric mean over all
    measurements); rows = one row per measurement with an 'organism' column (empty if the CSV has none)."""
    raw = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in raw.columns}
    if "sequence" not in cols:
        sys.exit("CSV needs a 'sequence' column.")
    seq = raw[cols["sequence"]].astype(str).str.strip().str.upper()
    mic_col = next((cols[c] for c in ("mic_um", "mic") if c in cols), None)
    org_col = (cols.get(organism_col.lower().strip()) if organism_col
               else next((cols[c] for c in ORG_COLS if c in cols), None))
    if organism_col and org_col is None:
        sys.exit(f"--organism_col '{organism_col}' not found. Columns: {list(raw.columns)}")
    rows = pd.DataFrame({"sequence": seq})
    rows["mic_um"] = pd.to_numeric(raw[mic_col], errors="coerce") if mic_col else np.nan
    rows["organism"] = raw[org_col].astype(str).str.strip().str.lower() if org_col else ""
    n0 = len(rows)
    rows = rows[rows.sequence.map(lambda q: is_valid(q, min_len, max_len))].reset_index(drop=True)
    # one row per sequence; multiple MIC measurements -> geometric mean
    out = rows.groupby("sequence", as_index=False)["mic_um"].agg(geo_mean)
    log(f"loaded {n0} rows -> {len(out)} unique valid sequences "
        f"({out.mic_um.notna().sum()} with MIC)"
        + (f"; organism column: '{org_col}'" if org_col else "; no organism column"))
    return out.reset_index(drop=True), rows


def select_organism(rows, organism):
    """Rows whose organism contains any comma-separated alias (case-insensitive). Returns one
    row per sequence (geometric-mean MIC over that organism's measurements)."""
    aliases = [a.strip().lower() for a in organism.split(",") if a.strip()]
    if not (rows.organism != "").any():
        sys.exit("--organism needs an organism column in the CSV (organism, bacterium, species, strain "
                 "or target; or name it with --organism_col). Rebuild your CSV with that column.")
    hit = rows[rows.organism.map(lambda o: any(a in o for a in aliases))]
    if hit.empty:
        top = rows.organism.value_counts().head(20)
        sys.exit(f"No rows match --organism '{organism}'. Most common values in the CSV:\n"
                 + "\n".join(f"  {n:>6}  {o}" for o, n in top.items())
                 + "\nUse one or more of these, comma-separated (e.g. --organism \"e. coli,escherichia coli\").")
    matched = sorted(hit.organism.unique())
    log(f"organism filter '{organism}': {len(hit)} measurements, {hit.sequence.nunique()} peptides; "
        f"matched labels: {matched[:8]}{' ...' if len(matched) > 8 else ''}")
    return hit.groupby("sequence", as_index=False)["mic_um"].agg(geo_mean)


def make_demo_data(n=3000, seed=0):
    """Synthetic cationic/amphipathic peptides. Only for checking the pipeline runs."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        L = rng.randint(10, 30)
        s = "".join(rng.choices("KKRRLLIVFWAGSTQ", k=L))
        mic = 2 ** (7 - 0.4 * net_charge(s) + rng.gauss(0, 1))
        rows.append((s, max(mic, 0.5)))
    return pd.DataFrame(rows, columns=["sequence", "mic_um"]).drop_duplicates("sequence")


# ----------------------------------------------------------------------- features
def net_charge(s):
    return s.count("K") + s.count("R") - s.count("D") - s.count("E")


def shannon(s):
    c, n = Counter(s), len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def has_repeat(s):
    return bool(re.search(r"(.)\1{3,}", s) or re.search(r"(.{2,3})\1{2,}", s))


def selfrep_frac(s, k=5):
    """Share of a sequence's k-mers that are repeats of an earlier k-mer in the same sequence.
    Catches long and imperfect internal repeats (e.g. a 12-residue unit tandem-repeated)."""
    km = [s[i:i + k] for i in range(len(s) - k + 1)]
    return 1 - len(set(km)) / len(km) if km else 0.0


def hydrophobic_moment(s, angle=100.0):
    a = math.radians(angle)
    h = [KD[c] for c in s]
    x = sum(v * math.cos(a * i) for i, v in enumerate(h))
    y = sum(v * math.sin(a * i) for i, v in enumerate(h))
    return math.hypot(x, y) / len(s)


def features(seqs):
    rows = []
    for s in seqs:
        comp = [s.count(a) / len(s) for a in AA]
        rows.append(comp + [len(s), net_charge(s), np.mean([KD[c] for c in s]),
                            hydrophobic_moment(s)])
    return np.array(rows, dtype=np.float32)


# ------------------------------------------------------------------------ model
def build_model(hidden, layers, dropout):
    import torch.nn as nn

    class LSTMLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, 128, padding_idx=PAD)
            self.lstm = nn.LSTM(128, hidden, layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden, VOCAB)

        def forward(self, x, state=None):
            h, state = self.lstm(self.emb(x), state)
            return self.out(self.drop(h)), state

    return LSTMLM()


def encode(seqs, max_len):
    import torch
    L = max_len + 2
    X = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids = [BOS] + [STOI[c] for c in s] + [EOS]
        X[i, :len(ids)] = torch.tensor(ids)
    return X


def train(seqs, args, device):
    import torch
    import torch.nn as nn
    X = encode(seqs, args.max_len)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(X), generator=g)
    n_val = max(1, int(0.05 * len(X)))
    Xva, Xtr = X[perm[:n_val]], X[perm[n_val:]]

    model = build_model(args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)
    best, bad = float("inf"), 0
    ckpt = os.path.join(args.out, "lstm_best.pt")

    def run(batch):
        batch = batch.to(device)
        logits, _ = model(batch[:, :-1])
        return lossf(logits.reshape(-1, VOCAB), batch[:, 1:].reshape(-1))

    for ep in range(1, args.epochs + 1):
        model.train()
        idx = torch.randperm(len(Xtr), generator=g)
        tot, nb = 0.0, 0
        for i in range(0, len(idx), args.batch):
            loss = run(Xtr[idx[i:i + args.batch]])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        model.eval()
        with torch.no_grad():
            val = np.mean([run(Xva[i:i + 1024]).item() for i in range(0, len(Xva), 1024)])
        log(f"epoch {ep:02d}  train {tot / nb:.3f}  val {val:.3f}")
        if val < best - 1e-4:
            best, bad = val, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                log("early stopping (validation loss stopped improving)")
                break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    with open(ckpt.replace(".pt", ".json"), "w") as f:
        json.dump(dict(hidden=args.hidden, layers=args.layers, dropout=args.dropout, seed=args.seed,
                       train_hash=getattr(args, "train_hash", None), n_train=len(seqs)), f)
    return model


def load_weights(path, args, device):
    """Load a trained generator. Returns (model, note). Uses the .json saved next to the weights
    (architecture + a fingerprint of the training set) when it exists."""
    import torch
    if not os.path.exists(path):
        sys.exit(f"--weights file not found: {path}")
    cfg_path = path.replace(".pt", ".json")
    note = f"generator: loaded weights from {path}"
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        for k in ("hidden", "layers", "dropout"):
            if k in cfg and getattr(args, k) != cfg[k]:
                log(f"using {k}={cfg[k]} from {cfg_path} (not {getattr(args, k)})")
                setattr(args, k, cfg[k])
        same = cfg.get("train_hash") in (None, args.train_hash)
        if not same:
            log("WARNING: these weights were trained on a different training set (different data, seed, "
                "--holdout or --train_cap_per_cluster). Novelty and held-out statistics in the report "
                "may be misleading, because the model may have seen peptides counted as held out.")
        note += (f" (training set matches this run: {'yes' if same else 'NO'})")
    else:
        log(f"no {os.path.basename(cfg_path)} next to the weights: assuming hidden={args.hidden}, "
            f"layers={args.layers}, dropout={args.dropout} and skipping the training-set check")
        note += " (no config file: architecture assumed, training-set check skipped)"
    model = build_model(args.hidden, args.layers, args.dropout).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError as e:
        sys.exit(f"could not load {path}: the architecture does not match (--hidden/--layers). {str(e)[:200]}")
    os.makedirs(args.out, exist_ok=True)
    for src in (path, cfg_path):  # keep the weights with this run's results
        dst = os.path.join(args.out, os.path.basename(src))
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)
    return model, note


def sample(model, n, args, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        while len(out) < n:
            b = min(args.gen_batch, n - len(out))
            x = torch.full((b, 1), BOS, dtype=torch.long, device=device)
            state, toks = None, []
            for t in range(args.max_len + 1):
                logits, state = model(x, state)
                logits = logits[:, -1] / args.temperature
                logits[:, [PAD, BOS]] = -float("inf")
                if t < args.min_len:
                    logits[:, EOS] = -float("inf")
                if t == args.max_len:  # force termination
                    logits[:] = -float("inf")
                    logits[:, EOS] = 0.0
                if args.top_k > 0:
                    kth = torch.topk(logits, args.top_k, dim=-1).values[:, -1:]
                    logits[logits < kth] = -float("inf")
                x = torch.multinomial(torch.softmax(logits, -1), 1)
                toks.append(x)
            arr = torch.cat(toks, 1).cpu().numpy()
            for row in arr:
                s = []
                for tok in row:
                    if tok == EOS:
                        break
                    s.append(ITOS[int(tok)])
                out.append("".join(s))
    return out


# ----------------------------------------------------------------------- filters
def filter_chain(raw, train_set, args, ref=None, max_overlap=None):
    """Sequential gates. Returns (survivors, list of (gate, n_remaining))."""
    steps = []
    cur = [s for s in raw if is_valid(s, args.min_len, args.max_len)]
    steps.append(("valid (20 aa, length range)", len(cur)))
    cur = list(dict.fromkeys(cur))
    steps.append(("unique within generated pool", len(cur)))
    cur = [s for s in cur if s not in train_set]
    steps.append(("not an exact training peptide", len(cur)))
    cur = [s for s in cur if shannon(s) >= args.min_entropy]
    steps.append((f"Shannon entropy >= {args.min_entropy}", len(cur)))
    cur = [s for s in cur if not has_repeat(s)]
    steps.append(("no homopolymer/tandem repeats", len(cur)))
    cur = [s for s in cur if selfrep_frac(s) <= args.max_selfrep]
    steps.append((f"internal 5-mer repeats <= {args.max_selfrep:.2f}", len(cur)))
    if args.no_cys:
        cur = [s for s in cur if "C" not in s]
        steps.append(("no cysteine", len(cur)))
    cur = [s for s in cur if net_charge(s) >= args.min_charge]
    steps.append((f"net charge >= +{args.min_charge}", len(cur)))
    if ref is not None and max_overlap is not None:
        cur = [s for s in cur if overlap_frac(s, ref) <= max_overlap]
        steps.append((f"5-mer overlap with training <= {max_overlap:.2f} (near-copy gate)", len(cur)))
    return cur, steps


# ----------------------------------------------------------------------- ranking
def cluster_seqs(seqs, k=3, thr=0.3):
    """Greedy leader clustering. A peptide joins the first-best leader it shares enough k-mers
    with (shared k-mers / k-mers of the shorter peptide >= thr); otherwise it starts a cluster.
    Returns a cluster id per sequence. Crude but fast, and needs no external tools."""
    from collections import defaultdict
    kms = [{q[i:i + k] for i in range(len(q) - k + 1)} for q in seqs]
    index = defaultdict(list)
    cid = [-1] * len(seqs)
    for i in sorted(range(len(seqs)), key=lambda j: (-len(seqs[j]), j)):
        cand = Counter()
        for km in sorted(kms[i]):          # sorted: set order changes between runs (hash randomisation)
            for j in index.get(km, ()):
                cand[j] += 1
        best, best_f = None, 0.0
        for j in sorted(cand):             # sorted: ties are broken by lowest index, deterministically
            f = cand[j] / max(1, min(len(kms[i]), len(kms[j])))
            if f >= thr and f > best_f:
                best, best_f = j, f
        if best is None:
            cid[i] = i
            for km in sorted(kms[i]):
                index[km].append(i)
        else:
            cid[i] = best
    return cid


def split_by_cluster(cid, test_frac, seed):
    """Whole clusters go to test, so no near-analog of a test peptide is in training."""
    groups = {}
    for i, c in enumerate(cid):
        groups.setdefault(c, []).append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    test = []
    for key in keys:
        if len(test) >= test_frac * len(cid):
            break
        test.extend(groups[key])
    test_set = set(test)
    train = [i for i in range(len(cid)) if i not in test_set]
    return train, sorted(test_set)


def _spearman(a, b):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    except ImportError:
        return float("nan")


def fit_ranker(train_df, seed, cluster_k=3, cluster_thr=0.3):
    """Predict log10(MIC in uM) from simple features. Lower = more potent.
    Evaluated on a random split AND a cluster split; the final model is refit on everything."""
    d = train_df[train_df.mic_um.notna() & (train_df.mic_um > 0)].reset_index(drop=True)
    if len(d) < 200:
        return None, None
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    seqs = d.sequence.tolist()
    X, y = features(seqs), np.log10(d.mic_um.values)

    def new_rf():
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    rho_random = _spearman(new_rf().fit(Xtr, ytr).predict(Xte), yte)

    cid = cluster_seqs(seqs, cluster_k, cluster_thr)
    n_clusters = len(set(cid))
    tr, te = split_by_cluster(cid, 0.2, seed)
    rho_cluster = _spearman(new_rf().fit(X[tr], y[tr]).predict(X[te]), y[te])
    log(f"ranker Spearman: random split {rho_random:.3f} | cluster split {rho_cluster:.3f} "
        f"({len(seqs)} peptides in {n_clusters} clusters)")
    info = dict(rho_random=rho_random, rho_cluster=rho_cluster, n=len(seqs), n_clusters=n_clusters)
    final = new_rf().fit(X, y)        # fitting is deterministic given random_state, even in parallel
    final.set_params(n_jobs=1)        # prediction is not: with threads the trees are summed in varying order,
    return final, info                # which changes scores in the last digits and can reorder near-ties


def score(seqs, ranker):
    """Higher = better candidate."""
    if ranker is not None:
        return np.round(-ranker.predict(features(seqs)), 9)
    # crude placeholder when there is no MIC data: cationic + amphipathic
    return np.round(np.array([hydrophobic_moment(s) + 0.5 * min(net_charge(s), 6) for s in seqs]), 9)


def identity(a, b):
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def pick_top(cand, cid, k=100, max_per_cluster=3, ident=0.7, neighbors=None, max_per_neighbor=0,
             max_family=0, fam_ident=0.6):
    """cand: sequences sorted best-first; cid: similarity-cluster id per candidate;
    neighbors: nearest known peptide per candidate (only used if max_per_neighbor > 0).
    Greedy: take the best candidates subject to a per-cluster cap, an optional per-neighbor cap, an optional
    cap on family size (picks linked at identity >= fam_ident form a family; chains count together)
    and a pairwise-identity cutoff. If that cannot fill k, constraints are relaxed in stages."""
    hi = max(ident, 0.85)
    # (identity cutoff, per-cluster cap, per-neighbor cap extra, family-size cap extra)
    stages = [(ident, max_per_cluster, 0, 0), (hi, max_per_cluster, 0, 0), (hi, max_per_cluster + 1, 1, 3),
              (hi, max_per_cluster + 2, 2, 6), (0.95, max_per_cluster + 5, 5, 12)]
    use_nb = max_per_neighbor > 0 and neighbors is not None
    use_fam = max_family > 0
    chosen, per, per_nb = [], Counter(), Counter()
    parent, size = {}, {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used = 0
    for st, (thr, cap, nb_extra, fam_extra) in enumerate(stages):
        nb_cap, fam_cap = max_per_neighbor + nb_extra, max_family + fam_extra
        for i in range(len(cand)):
            if len(chosen) == k:
                break
            if i in parent or per[cid[i]] >= cap:
                continue
            if use_nb and per_nb[neighbors[i]] >= nb_cap:
                continue
            ok, roots = True, set()
            for j in chosen:
                r = identity(cand[i], cand[j])
                if r >= thr:
                    ok = False
                    break
                if use_fam:
                    r = max(r, identity(cand[j], cand[i]))   # the ratio is not symmetric: use the larger
                    if r >= fam_ident:
                        roots.add(find(j))
            if not ok:
                continue
            if use_fam and 1 + sum(size[r] for r in roots) > fam_cap:
                continue
            chosen.append(i)
            per[cid[i]] += 1
            if use_nb:
                per_nb[neighbors[i]] += 1
            parent[i], size[i] = i, 1
            for r in roots:           # merge the families this pick links together
                parent[r] = i
                size[i] += size[r]
        used = st
        if len(chosen) == k:
            break
    note = None
    if used > 0:
        thr, cap, nb_extra, fam_extra = stages[used]
        extra = (f", max {max_per_neighbor + nb_extra} per known neighbor" if use_nb else "") + \
                (f", max family size {max_family + fam_extra}" if use_fam else "")
        note = (f"WARNING: could not fill top-{k} under the strict settings; relaxed to identity < {thr}, "
                f"max {cap} per cluster{extra} (stage {used + 1}/{len(stages)})")
    if len(chosen) < k:
        note = (note or "") + f" Only {len(chosen)} candidates available."
    return chosen, note


def family_sizes(seqs, rel_ident=0.6):
    """Sizes of 'families' among picks: connected components when peptides with
    identity >= rel_ident are linked (chains of variants end up together). Largest first."""
    n = len(seqs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i):
            if max(identity(seqs[i], seqs[j]), identity(seqs[j], seqs[i])) >= rel_ident:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return sorted(Counter(find(i) for i in range(n)).values(), reverse=True)


def build_kmer_index(seqs, k=3):
    from collections import defaultdict
    idx = defaultdict(list)
    for n, q in enumerate(seqs):
        for km in sorted({q[i:i + k] for i in range(len(q) - k + 1)}):
            idx[km].append(n)
    return idx


def nearest_known(q, known, idx, k=3, n_cand=30):
    """Most similar known peptide (difflib ratio) among the n_cand sharing the most 3-mers."""
    c = Counter()
    for km in sorted({q[i:i + k] for i in range(len(q) - k + 1)}):
        for n in idx.get(km, ()):
            c[n] += 1
    best_r, best = 0.0, ""
    for n, _ in sorted(c.items(), key=lambda t: (-t[1], t[0]))[:n_cand]:
        r = identity(q, known[n])
        if r > best_r:
            best_r, best = r, known[n]
    return best, best_r


# ------------------------------------------------------------------------- main
def kmer_set(seqs, k=5):
    return {s[i:i + k] for s in seqs for i in range(len(s) - k + 1)}


def overlap_frac(s, ref, k=5):
    """Fraction of a sequence's k-mers that also occur somewhere in the reference set."""
    km = [s[i:i + k] for i in range(len(s) - k + 1)]
    return sum(x in ref for x in km) / len(km) if km else 0.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data")
    p.add_argument("--demo", action="store_true", help="synthetic data, for smoke tests only")
    p.add_argument("--out", default="amp_out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_len", type=int, default=8)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--n_raw", type=int, default=300_000, help="sequences to sample before filtering")
    p.add_argument("--n_library", type=int, default=50_000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0, help="0 = off")
    p.add_argument("--gen_batch", type=int, default=2048)
    p.add_argument("--min_entropy", type=float, default=2.0)
    p.add_argument("--min_charge", type=int, default=2)
    p.add_argument("--no_cys", action="store_true", help="exclude cysteine (easier synthesis)")
    p.add_argument("--organism", default=None,
                   help="train the potency ranker only on MIC data for this organism, e.g. "
                        "\"e. coli,escherichia coli\" (comma-separated aliases, case-insensitive substring match). "
                        "The generator still trains on all peptides.")
    p.add_argument("--organism_col", default=None, help="name of the organism column (auto-detected if omitted)")
    p.add_argument("--list_organisms", action="store_true", help="print the organisms in the CSV with counts and exit")
    p.add_argument("--max_selfrep", type=float, default=0.2,
                   help="drop sequences whose share of repeated internal 5-mers exceeds this")
    p.add_argument("--max_per_cluster", type=int, default=3,
                   help="top-100: at most this many peptides from one similarity cluster")
    p.add_argument("--cluster_k", type=int, default=3, help="k-mer size for similarity clustering")
    p.add_argument("--cluster_thr", type=float, default=0.3,
                   help="clustering: shared k-mers / k-mers of the shorter peptide to join a cluster "
                        "(lower = bigger clusters)")
    p.add_argument("--top_ident", type=float, default=0.7,
                   help="top-100: max pairwise identity between picks (relaxed with a warning if needed)")
    p.add_argument("--max_known_identity", type=float, default=0.8,
                   help="top-100: exclude candidates whose most similar known (training or held-out) "
                        "peptide is at least this identical; use 1.1 to disable")
    p.add_argument("--max_examine", type=int, default=20000,
                   help="max number of candidates checked against known peptides when selecting top-100")
    p.add_argument("--train_cap_per_cluster", type=int, default=0,
                   help="family-balanced generator training: keep at most this many peptides per "
                        "similarity cluster (e.g. 5). 0 = use all. Copy checks still use ALL known peptides.")
    p.add_argument("--max_per_neighbor", type=int, default=0,
                   help="top-100: at most this many picks may share the same nearest known peptide (0 = off)")
    p.add_argument("--max_family", type=int, default=0,
                   help="top-100: maximum size of a 'family' of picks. Picks are linked when their identity "
                        "(difflib ratio) is >= --family_identity, and a family is a connected group of linked "
                        "picks, so chains of variants count together. Limits how much of the list one peptide "
                        "series can take (0 = off; try 8-12)")
    p.add_argument("--family_identity", type=float, default=0.6,
                   help="identity at which two top-100 picks count as relatives of the same family")
    p.add_argument("--weights", default=None,
                   help="skip training and sample from these saved generator weights (e.g. run_x/lstm_best.pt). "
                        "Use the same --data/--seed/--holdout/--train_cap_per_cluster as the run that made them")
    p.add_argument("--sample_seed", type=int, default=None,
                   help="seed used right before sampling (default: --seed + 1). Same weights + same sample seed "
                        "= same raw sequences")
    p.add_argument("--nondeterministic", action="store_true",
                   help="turn off PyTorch deterministic mode (slightly faster; runs will not repeat exactly)")
    p.add_argument("--top_pool", type=int, default=5000, help="top-100 is chosen from this many best-scored candidates")
    p.add_argument("--top_min_len", type=int, default=None, help="top-100 only: minimum length")
    p.add_argument("--top_max_len", type=int, default=None, help="top-100 only: maximum length (e.g. 30)")
    p.add_argument("--top_max_charge", type=int, default=None,
                   help="top-100 only: maximum net charge (e.g. 8). Very cationic peptides tend to be more "
                        "hemolytic, and the potency ranker tends to reward charge.")
    p.add_argument("--holdout", type=float, default=0.1,
                   help="fraction of real peptides kept out of training to measure how much "
                        "real, unseen AMPs overlap with the training set (0 disables)")
    p.add_argument("--max_overlap", type=float, default=None,
                   help="drop sequences whose 5-mer overlap with the training set exceeds this "
                        "(default: the --overlap_quantile of the held-out real peptides)")
    p.add_argument("--overlap_quantile", type=float, default=0.95,
                   help="quantile of held-out overlap used as the near-copy threshold; "
                        "lower = stricter (try 0.75)")
    args, _ = p.parse_known_args(argv)  # ignores Jupyter/Colab's extra '-f kernel.json'

    if not args.data and not args.demo:
        sys.exit("Give --data your.csv, or --demo for a smoke test.")
    if args.demo:
        log("DEMO MODE: synthetic data, outputs are not real AMP designs")
        args.n_raw, args.epochs, args.n_library = min(args.n_raw, 20_000), min(args.epochs, 3), 2_000
        args.hidden = min(args.hidden, 128)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed, not args.nondeterministic)

    if args.demo:
        if args.organism or args.list_organisms:
            sys.exit("--demo data has no organism column.")
        df, rows = make_demo_data(seed=args.seed), None
    else:
        df, rows = load_data(args.data, args.min_len, args.max_len, args.organism_col)
    if args.list_organisms:
        vc = rows.organism.value_counts()
        print(f"{len(vc)} organism labels:")
        for o, n in vc.head(60).items():
            print(f"  {n:>6}  {o or '(empty)'}")
        return
    rank_df = select_organism(rows, args.organism) if args.organism else df
    df = df[df.sequence.map(lambda s: is_valid(s, args.min_len, args.max_len))].reset_index(drop=True)
    all_seqs = df.sequence.tolist()
    train_set = set(all_seqs)  # every known peptide: exact matches are never novel

    # hold out some real peptides: the model never sees them, so their overlap with the
    # training set is the "normal" level for real, unseen AMPs
    idx = list(range(len(all_seqs)))
    random.Random(args.seed).shuffle(idx)
    n_hold = int(args.holdout * len(idx))
    hold_ids = set(idx[:n_hold])
    holdout = [all_seqs[i] for i in sorted(hold_ids)]
    train_full = [s for i, s in enumerate(all_seqs) if i not in hold_ids]
    ref = kmer_set(train_full)  # copy checks always use every known training peptide
    hold_ov = [overlap_frac(s, ref) for s in holdout]
    if args.max_overlap is not None:
        max_ov = args.max_overlap
    elif hold_ov:
        max_ov = float(np.quantile(hold_ov, args.overlap_quantile))
    else:
        max_ov = None
    train_seqs = train_full
    if args.train_cap_per_cluster > 0:
        tc = cluster_seqs(train_full, args.cluster_k, args.cluster_thr)
        order_t = list(range(len(train_full)))
        random.Random(args.seed).shuffle(order_t)
        per_t, keep = Counter(), []
        for i in order_t:
            if per_t[tc[i]] < args.train_cap_per_cluster:
                per_t[tc[i]] += 1
                keep.append(i)
        train_seqs = [train_full[i] for i in sorted(keep)]
        log(f"family-balanced training: {len(train_full)} peptides in {len(set(tc))} clusters -> "
            f"keeping {len(train_seqs)} (max {args.train_cap_per_cluster} per cluster)")
    log(f"training on {len(train_seqs)} peptides, {len(holdout)} held out; "
        f"near-copy threshold: {'off' if max_ov is None else round(max_ov, 3)}")

    args.train_hash = hashlib.md5("\n".join(sorted(train_seqs)).encode()).hexdigest()
    if args.weights:
        model, gen_note = load_weights(args.weights, args, device)
    else:
        model = train(train_seqs, args, device)
        gen_note = "generator: trained in this run"
    sample_seed = args.sample_seed if args.sample_seed is not None else args.seed + 1
    set_seed(sample_seed, not args.nondeterministic)       # sampling no longer depends on how training used the RNG
    gen_note += f"; sampling seed {sample_seed}; PyTorch deterministic mode {'off' if args.nondeterministic else 'on'}"
    log(f"sampling {args.n_raw} raw sequences")
    raw = sample(model, args.n_raw, args, device)

    surv, steps = filter_chain(raw, train_set, args, ref, max_ov)
    log(f"{len(surv)} sequences survived all gates")
    if len(surv) < args.n_library:
        log(f"WARNING: only {len(surv)} survivors for a library of {args.n_library}. "
            f"Raise --n_raw, lower filters, or raise --temperature.")
    rng = random.Random(args.seed)
    lib = rng.sample(surv, min(args.n_library, len(surv)))

    ranker, rinfo = fit_ranker(rank_df, args.seed, args.cluster_k, args.cluster_thr)
    if args.organism and ranker is None:
        log(f"WARNING: fewer than 200 peptides with MIC for '{args.organism}'; using the heuristic placeholder ranker")
    sc = score(lib, ranker)
    lo = args.top_min_len if args.top_min_len is not None else args.min_len
    hi = args.top_max_len if args.top_max_len is not None else args.max_len
    known_idx = build_kmer_index(all_seqs)
    nn_cache, order, examined, dropped = {}, [], 0, 0
    for i in np.argsort(-sc, kind="stable"):     # stable: tied scores keep library order
        if len(order) >= args.top_pool or examined >= args.max_examine:
            break
        if not (lo <= len(lib[i]) <= hi):
            continue
        if args.top_max_charge is not None and net_charge(lib[i]) > args.top_max_charge:
            continue
        if args.max_known_identity <= 1.0 or args.max_per_neighbor > 0:
            nn_cache[i] = nearest_known(lib[i], all_seqs, known_idx)
            examined += 1
            if args.max_known_identity <= 1.0 and nn_cache[i][1] >= args.max_known_identity:
                dropped += 1
                continue
        order.append(i)
    if args.max_known_identity <= 1.0:
        known_note = (f"known-peptide filter: excluded {dropped} of {examined} best-scored candidates whose "
                      f"nearest known peptide is >= {args.max_known_identity:.2f} identical")
        log(known_note)
    else:
        known_note = "known-peptide filter: off"
    if len(order) < 100:
        log(f"WARNING: only {len(order)} candidates left after the length/charge/known-peptide filters; "
            f"relax --top_max_charge / --top_max_len / --max_known_identity, or raise --max_examine / --n_raw")
    cand = [lib[i] for i in order]
    ccid = cluster_seqs(cand, args.cluster_k, args.cluster_thr)
    log(f"selecting top-100 from {len(cand)} candidates in {len(set(ccid))} similarity clusters")
    neighbors = [nn_cache[i][0] for i in order] if args.max_per_neighbor > 0 else None
    picked, pick_note = pick_top(cand, ccid, 100, args.max_per_cluster, args.top_ident,
                                 neighbors, args.max_per_neighbor, args.max_family, args.family_identity)
    if pick_note:
        log(pick_note)
    top_idx = [order[j] for j in picked]
    top_cid = [ccid[j] for j in picked]

    pd.DataFrame({"sequence": lib}).to_csv(os.path.join(args.out, "library.csv"), index=False)
    with open(os.path.join(args.out, "library.fasta"), "w") as f:
        for i, s in enumerate(lib):
            f.write(f">lib_{i}\n{s}\n")
    top_seqs = [lib[i] for i in top_idx]
    fam_sz = family_sizes(top_seqs, args.family_identity) if top_seqs else [0]
    nn = [nn_cache.get(i) or nearest_known(lib[i], all_seqs, known_idx) for i in top_idx]
    ref_all = kmer_set(all_seqs)
    top = pd.DataFrame({"rank": range(1, len(top_idx) + 1),
                        "sequence": top_seqs,
                        "score": [float(sc[i]) for i in top_idx],
                        "pred_mic_um": [10 ** -float(sc[i]) if ranker is not None else np.nan for i in top_idx],  # vs --organism if set
                        "length": [len(q) for q in top_seqs],
                        "net_charge": [net_charge(q) for q in top_seqs],
                        "cluster": top_cid,
                        "nearest_known": [x[0] for x in nn],
                        "nearest_identity": [round(x[1], 3) for x in nn],
                        "kmer_overlap_known": [round(overlap_frac(q, ref_all), 3) for q in top_seqs]})
    top.to_csv(os.path.join(args.out, "top100.csv"), index=False)

    # ---- report: the checks worth looking at before trusting a run
    uniq_raw = len(set(raw))
    train_only = set(train_full)
    in_train = sum(s in train_only for s in raw)
    hold_set = set(holdout)
    in_hold = sum(x in hold_set for x in raw)
    lens = [len(s) for s in lib] or [0]
    lib_ov = [overlap_frac(s, ref) for s in lib]
    hold_set = set(holdout)
    rediscovered = len(hold_set & set(raw))
    lines = [
        "AMP baseline report", "=" * 40,
        f"training peptides: {len(train_full)}" + (
            f"  (generator trained on {len(train_seqs)} after family balancing, max "
            f"{args.train_cap_per_cluster} per cluster)" if len(train_seqs) != len(train_full) else ""),
        gen_note,
        f"raw samples: {len(raw)}   unique: {uniq_raw} ({100 * uniq_raw / len(raw):.1f}%)",
        f"raw samples that exactly copy a TRAINING peptide (memorization): "
        f"{in_train} ({100 * in_train / len(raw):.1f}%)",
        f"raw samples that exactly equal a HELD-OUT real peptide (rediscovery): "
        f"{in_hold} ({100 * in_hold / len(raw):.1f}%)",
        "", "Gate survival:",
    ] + [f"  {name}: {n}" for name, n in steps] + [
        "", f"library size: {len(lib)}",
        f"library length mean/min/max: {np.mean(lens):.1f}/{min(lens)}/{max(lens)}",
        f"library mean net charge: {np.mean([net_charge(s) for s in lib]):.2f}",
        f"library 5-mer overlap with training set: mean {np.mean(lib_ov):.3f}, "
        f"median {np.median(lib_ov):.3f}, 95th pct {np.quantile(lib_ov, 0.95):.3f}",
        (f"held-out REAL peptides (n={len(holdout)}) 5-mer overlap with training set: "
         f"mean {np.mean(hold_ov):.3f}, median {np.median(hold_ov):.3f}, "
         f"95th pct {np.quantile(hold_ov, 0.95):.3f}   <- baseline for 'normal' novelty"
         if holdout else "held-out baseline: off (--holdout 0)"),
        (f"held-out real peptides regenerated exactly by the model (raw pool): "
         f"{rediscovered} of {len(holdout)}" if holdout else "held-out rediscovery: off"),
        (f"ranker: random forest on {rinfo['n']} peptides"
         f"{' with MIC for organism ' + repr(args.organism) if args.organism else ''} in {rinfo['n_clusters']} similarity clusters. "
         f"Held-out Spearman: random split {rinfo['rho_random']:.3f}, cluster split "
         f"{rinfo['rho_cluster']:.3f} (the cluster split is the honest one)"
         if ranker else "ranker: heuristic placeholder (no/insufficient MIC data)"),
        f"top-100: chosen from the {len(cand)} best-scored library peptides (length {lo}-{hi}"
        f"{'' if args.top_max_charge is None else f', net charge <= {args.top_max_charge}'}), "
        f"max {args.max_per_cluster} per similarity cluster (k={args.cluster_k}, thr={args.cluster_thr}), "
        f"pairwise identity < {args.top_ident} (difflib ratio, not true Levenshtein)",
        known_note,
        pick_note or "top-100: constraints satisfied without relaxing",
        f"top-100 families (picks linked at identity >= {args.family_identity}): {len(fam_sz)} families, "
        f"largest has {fam_sz[0]} picks, top 5 sizes {fam_sz[:5]}"
        + (f"   [cap: max family size {args.max_family}]" if args.max_family > 0 else "   [no family cap]"),
        f"top-100: {len(set(top_cid))} distinct clusters; nearest-known-peptide identity median "
        f"{np.median(top['nearest_identity']):.2f}, {int((top['nearest_identity'] >= 0.8).sum())} picks >= 0.80 "
        f"(close analogs of a known peptide; see top100.csv columns nearest_known / nearest_identity)",
    ]
    def prop_row(name, seqs_):
        if not seqs_:
            return f"  {name:<22}n=0"
        F = features(seqs_)[:, -4:]  # length, net charge, mean hydropathy, hydrophobic moment
        m_, md = F.mean(0), np.median(F, 0)
        return (f"  {name:<22}n={len(seqs_):<6} length {m_[0]:5.1f} ({md[0]:.0f})   charge {m_[1]:5.2f} ({md[1]:.0f})"
                f"   hydropathy {m_[2]:5.2f} ({md[2]:.2f})   moment {m_[3]:5.2f} ({md[3]:.2f})")
    lines += ["", "Property comparison, mean (median):",
              prop_row("training (all)", train_full)]
    if len(train_seqs) != len(train_full):
        lines.append(prop_row("training (model input)", train_seqs))
    lines += [prop_row("held-out real", holdout),
              prop_row("generated library", lib),
              "  (a library far from the real peptides on these is a sign it is not AMP-like)"]

    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"done. files in {args.out}/")


if __name__ == "__main__":
    main()
