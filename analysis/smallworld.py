"""Is the DUS network a small world? Clustering and distance against null models.

Reviewer 1 objected, correctly, that showing the mean distance falls below the
chain expectation is nearly tautological, that a Watts--Strogatz claim needs
clustering measured against a degree-preserving null, and that clustering is not
obviously defined for a DAG. This script addresses all three.

Clustering in a DAG
-------------------
The usual clustering coefficient counts closed triangles, and a *cyclic*
triangle cannot exist in a DAG. A *transitive* triangle can: i->j, j->k and the
shortcut i->k are mutually consistent with acyclicity. We therefore use the
transitive clustering coefficient

    C = (number of 2-paths i->j->k that are closed by an edge i->k)
        / (number of 2-paths i->j->k)

which is the DAG-admissible member of the directed-triangle family and is
exactly the quantity a shortcut creates. C = 0 for a pure chain.

Null models
-----------
Both preserve the in- and out-degree sequence, as Watts--Strogatz requires.

  spine   The DUS network always contains the spine i -> i+1 (every activity has
          a predecessor one level below), so the spine is not a property that
          randomisation should be free to destroy. This null holds the spine
          fixed and rewires only the skip edges by directed double-edge swaps,
          preserving each level's skip in-degree and skip out-degree and the
          constraint j > i + 1. It asks whether the *placement* of the shortcuts
          is special, given how many each level has.

  forward Randomises the underlying activity network instead: double-edge swaps
          on the activity DAG that preserve every activity's in- and out-degree
          and keep every edge pointing forward in the observed topological
          order, hence acyclic. The DUS decomposition is then recomputed from
          scratch. It asks whether the whole DUS structure, including its depth,
          follows from the activity degree sequence alone.

Usage
-----
    python analysis/smallworld.py <raw_dir> [--sample N] [--reps R]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from psplib import topological_sort

HERE = Path(__file__).parent
RNG = np.random.default_rng(20260806)


# ---------------------------------------------------------------------------
# DUS network extraction
# ---------------------------------------------------------------------------

def dus_network(sources, targets):
    """Return (m, edge set of the DUS network, DUS index array, remapped edges)."""
    st = np.concatenate([sources, targets])
    _, inv = np.unique(st, return_inverse=True)
    n = len(sources)
    s, t = inv[:n].astype(np.int64), inv[n:].astype(np.int64)

    order, succ = topological_sort(s, t)
    if not order:
        return None
    n_nodes = int(max(s.max(), t.max())) + 1
    u = np.zeros(n_nodes, dtype=np.int64)
    for i in order:
        if i in succ:
            u[succ[i]] = np.maximum(u[succ[i]], u[i] + 1)
    m = int(u.max()) + 1
    edges = set(zip(u[s].tolist(), u[t].tolist()))
    edges.discard(None)
    return m, {(a, b) for a, b in edges if a != b}, u, (s, t)


def adjacency(m, edges):
    A = np.zeros((m, m), dtype=bool)
    for a, b in edges:
        A[a, b] = True
    return A


def transitive_clustering(A):
    """Fraction of directed 2-paths that are closed by a transitive shortcut."""
    P = A.astype(np.int64) @ A.astype(np.int64)   # P[i,k] = #{j : i->j->k}
    twopaths = P.sum()
    if twopaths == 0:
        return np.nan
    return float(P[A].sum() / twopaths)


def mean_distance_to_end(m, edges):
    """Mean shortest distance from each level to the final level."""
    succ = [[] for _ in range(m)]
    for a, b in edges:
        succ[a].append(b)
    dist = np.zeros(m, dtype=np.int64)
    for i in range(m - 1)[::-1]:
        if not succ[i]:
            return np.nan
        dist[i] = min(dist[j] for j in succ[i]) + 1
    return float(dist.mean())


# ---------------------------------------------------------------------------
# null model 1: rewire the skip edges of the DUS network, spine held fixed
# ---------------------------------------------------------------------------

def null_spine(m, edges, rng, n_swap_factor=20):
    spine = {(i, i + 1) for i in range(m - 1)} & edges
    skip = [e for e in edges if e[1] > e[0] + 1]
    if len(skip) < 2:
        return None
    skip = list(skip)
    present = set(skip)
    n_try = n_swap_factor * len(skip)
    for _ in range(n_try):
        i, j = rng.integers(0, len(skip), 2)
        if i == j:
            continue
        (a, b), (c, d) = skip[i], skip[j]
        # swap targets; keep every edge a forward skip and avoid multi-edges
        if d <= a + 1 or b <= c + 1:
            continue
        if (a, d) in present or (c, b) in present:
            continue
        present.discard((a, b)); present.discard((c, d))
        present.add((a, d)); present.add((c, b))
        skip[i] = (a, d); skip[j] = (c, b)
    return spine | present


# ---------------------------------------------------------------------------
# null model 2: degree-preserving forward rewiring of the activity DAG
# ---------------------------------------------------------------------------

def null_forward(s, t, rng, n_swap_factor=10):
    """Double-edge swaps preserving in/out degree and forward-ness in a fixed
    topological order, so the result is a DAG with the same degree sequence."""
    n_nodes = int(max(s.max(), t.max())) + 1
    order, _ = topological_sort(s, t)
    if not order:
        return None
    pos = np.empty(n_nodes, dtype=np.int64)
    pos[np.asarray(order)] = np.arange(len(order))
    # isolated-from-order nodes (targets only) get positions too
    missing = np.setdiff1d(np.arange(n_nodes), np.asarray(order))
    if len(missing):
        pos[missing] = len(order) + np.arange(len(missing))

    S, T = s.copy(), t.copy()
    present = set(zip(S.tolist(), T.tolist()))
    E = len(S)
    for _ in range(n_swap_factor * E):
        i, j = rng.integers(0, E, 2)
        if i == j:
            continue
        a, b = S[i], T[i]
        c, d = S[j], T[j]
        if a == d or c == b:
            continue
        if pos[a] >= pos[d] or pos[c] >= pos[b]:      # keep edges forward
            continue
        if (a, d) in present or (c, b) in present:
            continue
        present.discard((a, b)); present.discard((c, d))
        present.add((a, d)); present.add((c, b))
        T[i], T[j] = d, b
    return S, T


# ---------------------------------------------------------------------------

def observed_stats(s, t):
    r = dus_network(s, t)
    if r is None:
        return None
    m, edges, u, _ = r
    if m < 4:
        return None
    A = adjacency(m, edges)
    return dict(m=m, n_dus_edges=len(edges),
                n_skip=sum(1 for a, b in edges if b > a + 1),
                C=transitive_clustering(A), L=mean_distance_to_end(m, edges))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--sample", type=int, default=200,
                    help="projects for the spine null")
    ap.add_argument("--forward-sample", type=int, default=40,
                    help="projects for the (expensive) activity-level null")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--forward-reps", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "smallworld.csv"))
    a = ap.parse_args()

    raw = Path(a.raw_dir)
    refs = sorted({p.name.split("__")[0]
                   for p in raw.glob("*__nl_pcore_task_rel.parquet")})

    # stratify by size so the whole range is represented
    sizes = []
    for v in refs:
        rel = pd.read_parquet(raw / f"{v}__nl_pcore_task_rel.parquet",
                              columns=["pred_task_id"])
        sizes.append((v, len(rel)))
    sz = pd.DataFrame(sizes, columns=["ref", "n"]).sort_values("n")
    sz["bin"] = pd.qcut(np.log10(sz.n.clip(lower=1)), 8, duplicates="drop")
    per = max(1, a.sample // sz.bin.nunique())
    sel = (sz.groupby("bin", observed=True)
             .apply(lambda g: g.sample(min(per, len(g)), random_state=1))
             .reset_index(drop=True))
    print(f"{len(refs)} projects available; {len(sel)} sampled for the null models")

    rows = []
    for k, (v, _n, _b) in enumerate(sel.itertuples(index=False), 1):
        task = pd.read_parquet(raw / f"{v}__nl_pcore_task.parquet",
                               columns=["task_id", "nl_meta_type"])
        rel = pd.read_parquet(raw / f"{v}__nl_pcore_task_rel.parquet",
                              columns=["pred_task_id", "succ_task_id"])
        task = task.loc[task.nl_meta_type != "L"]
        rel = rel.loc[rel.pred_task_id.isin(task.task_id)
                      & rel.succ_task_id.isin(task.task_id)]
        if len(rel) < 20:
            continue
        obs = observed_stats(rel.pred_task_id.values, rel.succ_task_id.values)
        if obs is None or not np.isfinite(obs["C"]):
            continue

        r = dus_network(rel.pred_task_id.values, rel.succ_task_id.values)
        m, edges, _u, (s, t) = r

        Cs, Ls = [], []
        for _ in range(a.reps):
            e2 = null_spine(m, edges, RNG)
            if e2 is None:
                continue
            Cs.append(transitive_clustering(adjacency(m, e2)))
            Ls.append(mean_distance_to_end(m, e2))
        obs.update(version_ref=v, n_dependencies=len(rel),
                   C_spine=np.nanmean(Cs) if Cs else np.nan,
                   L_spine=np.nanmean(Ls) if Ls else np.nan,
                   C_spine_sd=np.nanstd(Cs) if Cs else np.nan,
                   L_spine_sd=np.nanstd(Ls) if Ls else np.nan)
        rows.append(obs)
        if k % 25 == 0:
            print(f"  spine null {k}/{len(sel)}")

    d = pd.DataFrame(rows)

    # expensive activity-level null on a subset spanning the range
    sub = d.sort_values("n_dependencies")
    idx = np.unique(np.linspace(0, len(sub) - 1, min(a.forward_sample, len(sub))).astype(int))
    fwd = {}
    for k, v in enumerate(sub.iloc[idx].version_ref, 1):
        task = pd.read_parquet(raw / f"{v}__nl_pcore_task.parquet",
                               columns=["task_id", "nl_meta_type"])
        rel = pd.read_parquet(raw / f"{v}__nl_pcore_task_rel.parquet",
                              columns=["pred_task_id", "succ_task_id"])
        task = task.loc[task.nl_meta_type != "L"]
        rel = rel.loc[rel.pred_task_id.isin(task.task_id)
                      & rel.succ_task_id.isin(task.task_id)]
        st = np.concatenate([rel.pred_task_id.values, rel.succ_task_id.values])
        _, inv = np.unique(st, return_inverse=True)
        n = len(rel)
        s, t = inv[:n].astype(np.int64), inv[n:].astype(np.int64)
        Cs, Ls, Ms = [], [], []
        for _ in range(a.forward_reps):
            out = null_forward(s, t, RNG)
            if out is None:
                continue
            o = observed_stats(*out)
            if o and np.isfinite(o["C"]):
                Cs.append(o["C"]); Ls.append(o["L"]); Ms.append(o["m"])
        if Cs:
            fwd[v] = dict(C_fwd=float(np.mean(Cs)), L_fwd=float(np.mean(Ls)),
                          m_fwd=float(np.mean(Ms)))
        print(f"  forward null {k}/{len(idx)}")

    for col in ["C_fwd", "L_fwd", "m_fwd"]:
        d[col] = d.version_ref.map(lambda v: fwd.get(v, {}).get(col, np.nan))

    d.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(d)} projects, {len(fwd)} with the activity-level null")

    def rep(x):
        x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
        return f"median {np.median(x):.4f}  IQR [{np.percentile(x,25):.4f}, {np.percentile(x,75):.4f}]  n={len(x)}"

    print("\n=== observed ===")
    print("  C  ", rep(d.C)); print("  L  ", rep(d.L))
    print("=== spine-preserving degree null ===")
    print("  C  ", rep(d.C_spine)); print("  L  ", rep(d.L_spine))
    print("  C/C_null ", rep(d.C / d.C_spine)); print("  L/L_null ", rep(d.L / d.L_spine))
    ok = np.isfinite(d.C_fwd)
    if ok.any():
        print("=== activity-level degree null ===")
        print("  C  ", rep(d.C_fwd[ok])); print("  L  ", rep(d.L_fwd[ok]))
        print("  C/C_null ", rep((d.C / d.C_fwd)[ok]))
        print("  L/L_null ", rep((d.L / d.L_fwd)[ok]))
        print("  m/m_null ", rep((d.m / d.m_fwd)[ok]))

    (HERE / "smallworld_summary.json").write_text(json.dumps({
        "n_projects": int(len(d)),
        "C_obs_median": float(np.nanmedian(d.C)),
        "L_obs_median": float(np.nanmedian(d.L)),
        "C_spine_median": float(np.nanmedian(d.C_spine)),
        "L_spine_median": float(np.nanmedian(d.L_spine)),
        "C_ratio_spine_median": float(np.nanmedian(d.C / d.C_spine)),
        "L_ratio_spine_median": float(np.nanmedian(d.L / d.L_spine)),
        "n_forward": int(np.isfinite(d.C_fwd).sum()),
        "C_fwd_median": float(np.nanmedian(d.C_fwd)),
        "L_fwd_median": float(np.nanmedian(d.L_fwd)),
        "C_ratio_fwd_median": float(np.nanmedian(d.C / d.C_fwd)),
        "L_ratio_fwd_median": float(np.nanmedian(d.L / d.L_fwd)),
        "m_ratio_fwd_median": float(np.nanmedian(d.m / d.m_fwd)),
    }, indent=2))


if __name__ == "__main__":
    main()
