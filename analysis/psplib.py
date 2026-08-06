"""Replication of the DUS/DPS scaling analysis on public PSPLIB instances.

Addresses Reviewer 1 point 4 (replicate key results on a publicly available
scheduling dataset).

Downloads the four single-mode PSPLIB sets (j30, j60, j90, j120; 2040 instances
in total), extracts the precedence DAG of each instance, computes the DUS and
DPS decompositions with the same code path used for the proprietary schedules,
and refits the two scaling relations.

Run:  python analysis/psplib.py
Outputs: analysis/psplib_instances.csv, analysis/psplib_results.json, fig6.pdf
"""

import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).parent
REPO = HERE.parent
CACHE = HERE / "psplib_cache"
SETS = ["j30", "j60", "j90", "j120"]
URL = "https://www.om-db.wi.tum.de/psplib/download_dataset.php?set={s}&mode=sm&format=tgz"

RNG = np.random.default_rng(20260806)


# ---------------------------------------------------------------------------
# DUS / DPS computation (same definitions as the manuscript)
# ---------------------------------------------------------------------------

def topological_sort(sources, targets):
    """Kahn's algorithm. Returns (order, out-neighbour map)."""
    argsort = sources.argsort()
    sources = sources[argsort]
    targets = targets[argsort]
    unique_sources, indices = np.unique(sources, return_index=True)
    out_neighbours = dict(zip(unique_sources, np.split(targets, indices[1:])))

    unique_targets, counts = np.unique(targets, return_counts=True)
    in_degree = dict(zip(unique_targets, counts))
    n_targets = len(unique_targets)

    seed = list(set(unique_sources) - set(unique_targets))
    order = []
    while seed:
        s = seed.pop(0)
        order.append(s)
        for t in out_neighbours.get(s, []):
            in_degree[t] -= 1
            if in_degree[t] == 0:
                seed.append(t)
                n_targets -= 1
    return ([] if n_targets > 0 else order), out_neighbours


def dus_dps(sources, targets):
    """Return summary statistics of the DUS and DPS decompositions of a DAG."""
    st = np.concatenate([sources, targets])
    _, inv = np.unique(st, return_inverse=True)
    n_src = len(sources)
    s, t = inv[:n_src], inv[n_src:]
    n_nodes = int(max(s.max(), t.max())) + 1

    order, succ = topological_sort(s, t)
    if not order:
        return None

    # DUS index: u_i = 1 + max over predecessors
    u = np.zeros(n_nodes, dtype=np.int64)
    for i in order:
        if i in succ:
            u[succ[i]] = np.maximum(u[succ[i]], u[i] + 1)
    m = int(u.max()) + 1

    # predecessor DUS-level sets -> DPS labels; DUS network edge weights
    pred_levels = [set() for _ in range(n_nodes)]
    dus_edges = {}
    dus_succ = [set() for _ in range(m)]
    for i in order:
        if i in succ:
            gi = u[i]
            for j in succ[i]:
                gj = u[j]
                pred_levels[j].add(int(gi))
                dus_succ[gi].add(int(gj))
                dus_edges[(int(gi), int(gj))] = dus_edges.get((int(gi), int(gj)), 0) + 1

    dps_labels = {frozenset(p) for p in pred_levels}

    # mean shortest distance from each DUS level to the last level
    dist = np.zeros(m, dtype=np.int64)
    for i in range(m - 1)[::-1]:
        dist[i] = min(dist[k] for k in dus_succ[i]) + 1
    return dict(
        n_activities=n_nodes,
        n_dependencies=len(s),
        n_dus=m,
        n_dps=len(dps_labels),
        n_dus_edges=len(dus_edges),
        dus_distance_to_end=float(np.mean(dist)),
    )


# ---------------------------------------------------------------------------
# PSPLIB parsing
# ---------------------------------------------------------------------------

def fetch(setname):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{setname}.sm.tgz"
    if not p.exists():
        with urllib.request.urlopen(URL.format(s=setname), timeout=300) as r:
            p.write_bytes(r.read())
    return p


def parse_sm(text):
    """Extract the precedence relations of a PSPLIB .sm instance.

    The PRECEDENCE RELATIONS block lists, for each job, its number of successors
    followed by their job numbers.
    """
    lines = text.splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if "PRECEDENCE RELATIONS" in l)
    except StopIteration:
        return None
    i += 2  # skip the column-header line
    src, dst = [], []
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("*"):
            break
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            break
        job = int(parts[0])
        nsucc = int(parts[2])
        for s in parts[3:3 + nsucc]:
            src.append(job)
            dst.append(int(s))
        i += 1
    if not src:
        return None
    return np.array(src), np.array(dst)


def collect():
    rows = []
    for setname in SETS:
        p = fetch(setname)
        with tarfile.open(p, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".sm"):
                    continue
                raw = tf.extractfile(member).read().decode("latin-1")
                pr = parse_sm(raw)
                if pr is None:
                    continue
                stat = dus_dps(*pr)
                if stat is None:
                    continue
                stat["dataset"] = setname
                stat["instance"] = Path(member.name).name
                rows.append(stat)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# fitting (mirrors analysis/stats.py: multiplicative errors, bootstrap CIs)
# ---------------------------------------------------------------------------

def fit_mult(f, x, y, p0):
    g = lambda xx, *p: np.log(np.clip(f(xx, *p), 1e-12, None))
    popt, _ = optimize.curve_fit(g, x, np.log(y), p0=p0, maxfev=400000)
    resid = np.log(y) - g(x, *popt)
    n = len(x)
    s2 = np.mean(resid**2)
    ll = -0.5 * n * (np.log(2 * np.pi * s2) + 1) - np.sum(np.log(y))
    return popt, float(ll), resid


def boot(f, x, y, p0, nboot=1000):
    n, out = len(x), []
    for _ in range(nboot):
        idx = RNG.integers(0, n, n)
        try:
            popt, _, _ = fit_mult(f, x[idx], y[idx], p0)
            out.append(popt)
        except Exception:
            continue
    return np.percentile(np.asarray(out), [2.5, 97.5], axis=0).T


def compare(x, y, name):
    forms = {
        "log1": (lambda z, p: p * np.log(z), [3.0], 1),
        "log2": (lambda z, p, c: p * np.log(z) + c, [3.0, 1.0], 2),
        "power": (lambda z, a, b: a * z**b, [1.0, 0.3], 2),
        "linear": (lambda z, a, b: a * z + b, [0.05, 1.0], 2),
    }
    res = {}
    for k, (f, p0, npar) in forms.items():
        popt, ll, resid = fit_mult(f, x, y, p0)
        res[k] = dict(
            params=[float(v) for v in popt],
            ci=[[float(a), float(b)] for a, b in boot(f, x, y, p0)],
            r2=float(1 - np.sum(resid**2) / np.sum((np.log(y) - np.log(y).mean()) ** 2)),
            aic=float(2 * (npar + 1) - 2 * ll),
        )
    best = min(res, key=lambda k: res[k]["aic"])
    for k in res:
        res[k]["daic"] = res[k]["aic"] - res[best]["aic"]
    res["_best"] = best
    res["_n"] = int(len(x))
    res["_label"] = name
    return res


def public_examples():
    """The two anonymised real schedules shipped with the public code repository.

    These extend the publicly reproducible range from ~2.6e2 to ~1.3e5
    dependencies, three orders of magnitude beyond the largest PSPLIB instance.
    """
    base = REPO / "examples" / "data"
    rows = []
    for i in [1, 2]:
        f = base / f"task_rel_{i}.parquet"
        if not f.exists():
            continue
        rel = pd.read_parquet(f)
        stat = dus_dps(rel.pred_task_id.to_numpy(), rel.succ_task_id.to_numpy())
        if stat:
            stat["dataset"] = "repo_example"
            stat["instance"] = f"task_rel_{i}"
            rows.append(stat)
    return pd.DataFrame(rows)


def main():
    csv = HERE / "psplib_instances.csv"
    if csv.exists():
        d = pd.read_csv(csv)
    else:
        d = collect()
        d.to_csv(csv, index=False)
    print(f"instances: {len(d)}")
    print(d.groupby("dataset")[["n_activities", "n_dependencies", "n_dus", "n_dps"]]
          .agg(["min", "median", "max"]).to_string())

    x = d.n_dependencies.to_numpy(float)
    out = {"n_instances": int(len(d))}

    print("\n=== PSPLIB: number of DUS levels vs dependencies ===")
    r1 = compare(x, d.n_dus.to_numpy(float), "n_DUS")
    for k in ["log1", "log2", "power", "linear"]:
        v = r1[k]
        print(f"  {k:7s} R2={v['r2']:6.3f} dAIC={v['daic']:8.1f} "
              f"params={np.round(v['params'], 4).tolist()} CI={np.round(v['ci'], 4).tolist()}")
    print("  best:", r1["_best"])
    out["dus_scaling"] = r1

    print("\n=== PSPLIB: number of DPS classes vs dependencies ===")
    r2 = compare(x, d.n_dps.to_numpy(float), "n_DPS")
    for k in ["log1", "log2", "power", "linear"]:
        v = r2[k]
        print(f"  {k:7s} R2={v['r2']:6.3f} dAIC={v['daic']:8.1f} "
              f"params={np.round(v['params'], 4).tolist()} CI={np.round(v['ci'], 4).tolist()}")
    print("  best:", r2["_best"])
    out["dps_scaling"] = r2

    print("\n=== PSPLIB: mean DUS distance to end ===")
    m = d.n_dus.to_numpy(float)
    L = d.dus_distance_to_end.to_numpy(float)
    ratio = L / ((m - 1) / 2)
    rho, p = stats.spearmanr(m, L)
    out["distance"] = dict(
        spearman_rho=float(rho), spearman_p=float(p),
        L_median=float(np.median(L)),
        L_iqr=[float(np.percentile(L, 25)), float(np.percentile(L, 75))],
        m_median=float(np.median(m)),
        ratio_median=float(np.median(ratio)),
        frac_below_chain=float(np.mean(ratio < 1)),
    )
    print(" ", {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in out["distance"].items()})

    ex = public_examples()
    if len(ex):
        print("\n=== public example schedules from the code repository ===")
        print(ex[["instance", "n_activities", "n_dependencies", "n_dus", "n_dps",
                  "dus_distance_to_end"]].to_string(index=False))
        ex.to_csv(HERE / "public_examples.csv", index=False)
        out["public_examples"] = ex.to_dict(orient="records")

    (HERE / "psplib_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote analysis/psplib_instances.csv and analysis/psplib_results.json")


if __name__ == "__main__":
    main()
