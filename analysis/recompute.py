"""Recompute the per-project summary statistics from raw schedules, exactly.

Why this exists
---------------
The summary table used for the first version of the analysis was produced by a
pipeline that built the DPS label with ``math.pow``, i.e. in float64. A float64
mantissa holds 53 bits, so summing ``2 ** j`` as floats silently merges distinct
predecessor patterns once a schedule has more than 53 DUS levels, and the count
of distinct DPS classes is then an under-estimate that grows with depth. On the
construction schedules the under-count was zero at or below 53 levels, a median
of 4.6% between 100 and 200 levels, and 34% above 200 levels, reaching 1.51x.

This script recomputes every quantity with exact integer arithmetic. The DUS
index, the DUS network and the distances are unaffected by the bug and are
reproduced identically; only the DPS counts change.

Input
-----
A directory of parquet files named

    {version_ref}__nl_pcore_task.parquet       (task_id, nl_meta_type, physical_complete_pct)
    {version_ref}__nl_pcore_task_rel.parquet   (pred_task_id, succ_task_id)

one pair per project. Activities with ``nl_meta_type == 'L'`` are dropped and
dependencies are restricted to the surviving activities, matching the original
pipeline. The schedules are proprietary and are not distributed with this
repository.

Usage
-----
    python analysis/recompute.py <raw_dir> <out_csv> [--index index.csv]

``--index`` optionally supplies a CSV with ``version_ref`` and ``tenant_ref``
columns so the company grouping is carried through; without it the company
column is left blank.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from psplib import topological_sort


def analyse(sources, targets):
    """Exact DUS/DPS statistics of a dependency DAG.

    Returns None if the network is cyclic.
    """
    st = np.concatenate([sources, targets])
    _, inv = np.unique(st, return_inverse=True)
    n = len(sources)
    s, t = inv[:n], inv[n:]
    n_nodes = int(max(s.max(), t.max())) + 1

    order, succ = topological_sort(s, t)
    if not order:
        return None

    u = np.zeros(n_nodes, dtype=np.int64)
    for i in order:
        if i in succ:
            u[succ[i]] = np.maximum(u[succ[i]], u[i] + 1)
    m = int(u.max()) + 1

    pred_levels = [0] * n_nodes          # exact integer bitmasks, not floats
    dus_edges = {}
    dus_succ = [set() for _ in range(m)]
    for i in order:
        if i in succ:
            gi = int(u[i])
            for j in succ[i]:
                gj = int(u[j])
                pred_levels[j] |= 1 << gi
                dus_succ[gi].add(gj)
                dus_edges[(gi, gj)] = dus_edges.get((gi, gj), 0) + 1

    dist = np.zeros(m, dtype=np.int64)
    for i in range(m - 1)[::-1]:
        dist[i] = min(dist[k] for k in dus_succ[i]) + 1

    skip = sum(1 for (a, b) in dus_edges if b > a + 1)
    return dict(
        n_nodes_in_graph=n_nodes,
        n_dependencies=len(s),
        n_generations=m,
        n_fibrations=len(set(pred_levels)),
        n_dus_edges=len(dus_edges),
        n_dus_skip_edges=skip,
        generation_average_distance_to_end=float(np.mean(dist)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--index", default=None)
    a = ap.parse_args()

    raw = Path(a.raw_dir)
    refs = sorted({p.name.split("__")[0]
                   for p in raw.glob("*__nl_pcore_task_rel.parquet")})
    tenants = {}
    if a.index:
        idx = pd.read_csv(a.index)
        tenants = dict(zip(idx.version_ref, idx.tenant_ref))

    rows, skipped = [], []
    for k, v in enumerate(refs, 1):
        try:
            task = pd.read_parquet(raw / f"{v}__nl_pcore_task.parquet",
                                   columns=["task_id", "nl_meta_type"])
            rel = pd.read_parquet(raw / f"{v}__nl_pcore_task_rel.parquet",
                                  columns=["pred_task_id", "succ_task_id"])
        except Exception as e:
            skipped.append((v, f"read: {e}")); continue

        task = task.loc[task.nl_meta_type != "L"]
        rel = rel.loc[rel.pred_task_id.isin(task.task_id)
                      & rel.succ_task_id.isin(task.task_id)]
        if len(rel) == 0:
            skipped.append((v, "no dependencies")); continue

        res = analyse(rel.pred_task_id.values, rel.succ_task_id.values)
        if res is None:
            skipped.append((v, "cyclic")); continue

        res["version_ref"] = v
        res["tenant_ref"] = tenants.get(v, "")
        res["n_activities"] = len(task)
        rows.append(res)
        if k % 200 == 0:
            print(f"  {k}/{len(refs)}")

    d = pd.DataFrame(rows)
    cols = ["tenant_ref", "version_ref", "n_activities", "n_dependencies",
            "n_generations", "n_fibrations", "n_dus_edges", "n_dus_skip_edges",
            "generation_average_distance_to_end", "n_nodes_in_graph"]
    d[cols].to_csv(a.out_csv, index=False)
    print(f"wrote {a.out_csv}: {len(d)} projects, {len(skipped)} skipped")
    for v, why in skipped[:10]:
        print(f"  skipped {v}: {why}")


if __name__ == "__main__":
    main()
