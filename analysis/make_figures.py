"""Generate all manuscript figures.

fig1  schematic: WBS, dependency network, DUS and DPS (new Figure 1)
fig2  DUS profiles of two real construction schedules
fig3  number of DUS levels vs number of dependencies, with fit and CI band
fig4  mean DUS distance to the end vs number of DUS levels
fig5  number of DPS classes vs number of dependencies, with fit and CI band
fig6  PSPLIB replication

Run:  python analysis/make_figures.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

HERE = Path(__file__).parent
REPO = HERE.parent

sys.path.insert(0, str(REPO))     # the dus package
sys.path.insert(0, str(HERE))     # stats.py

from stats import load, COMPANIES  # noqa: E402

OUT = REPO / "manuscript"
EXAMPLES = REPO / "examples" / "data"

CB = {"blue": "#377eb8", "orange": "#ff7f00", "green": "#4daf4a"}
RES = json.loads((HERE / "results.json").read_text())

plt.rcParams.update({"font.size": 13})


def savefig(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white",
                edgecolor="none", dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Figure 1 -- schematic (Reviewer 2, Reviewer 3 points 1, 2, 6)
# ---------------------------------------------------------------------------

# A seven-activity toy project used throughout Figure 1.
WBS_EDGES = [
    ("House", "Substructure"), ("House", "Superstructure"), ("House", "Services"),
    ("Substructure", "Excavate"), ("Substructure", "Foundations"),
    ("Superstructure", "Frame A"), ("Superstructure", "Frame B"),
    ("Superstructure", "Roof"),
    ("Services", "Utilities"), ("Services", "Fit-out"),
]
DEPS = [
    ("Excavate", "Foundations"),
    ("Foundations", "Frame A"), ("Foundations", "Frame B"),
    ("Excavate", "Utilities"), ("Foundations", "Utilities"),
    ("Frame A", "Roof"), ("Frame B", "Roof"),
    ("Frame A", "Fit-out"), ("Utilities", "Fit-out"),
]
PALETTE = ["#bdbdbd", "#80b1d3", "#8dd3c7", "#fdb462", "#bc80bd"]


def _toy_labels():
    """DUS index and DPS class (set of predecessor DUS levels) of the toy project."""
    preds = {}
    for s, t in DEPS:
        preds.setdefault(t, []).append(s)
    nodes = ["Excavate", "Foundations", "Frame A", "Frame B", "Utilities",
             "Roof", "Fit-out"]
    u = {}
    for n in nodes:                       # nodes are already in topological order
        u[n] = 0 if n not in preds else 1 + max(u[p] for p in preds[n])
    pl = {n: frozenset(u[p] for p in preds.get(n, [])) for n in nodes}
    order = sorted(set(pl.values()), key=lambda s: (len(s), sorted(s)))
    cls = {c: i for i, c in enumerate(order)}
    return nodes, u, pl, cls


def fig1():
    nodes, u, pl, cls = _toy_labels()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4),
                           gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0]))

    # --- (A) Work Breakdown Structure ---------------------------------------
    T = nx.DiGraph()
    T.add_edges_from(WBS_EDGES)
    pos = {
        "House": (3.0, 2),
        "Substructure": (0.7, 1), "Superstructure": (3.0, 1), "Services": (5.3, 1),
        "Excavate": (0.0, 0), "Foundations": (1.4, 0),
        "Frame A": (2.6, 0), "Frame B": (3.6, 0), "Roof": (4.5, 0),
        "Utilities": (5.5, 0), "Fit-out": (6.4, 0),
    }
    nx.draw_networkx_edges(T, pos, ax=ax[0], arrows=False, edge_color="0.6")
    for n, (px, py) in pos.items():
        leaf = T.out_degree(n) == 0
        ax[0].text(px, py - (0.22 if leaf else 0), n, ha="center", va="center",
                   fontsize=8.5, rotation=45 if leaf else 0,
                   rotation_mode="anchor" if leaf else None,
                   bbox=dict(boxstyle="round,pad=0.28",
                             fc=PALETTE[cls[pl[n]]] if leaf else "0.93", ec="0.45"))
    ax[0].set_title("(A) Work Breakdown Structure", loc="left", fontsize=13)
    ax[0].set(xlim=(-1.2, 7.6), ylim=(-1.5, 2.45))
    ax[0].axis("off")

    # --- (B) dependency network of the WBS leaves ----------------------------
    xy = {
        "Excavate": (0, 0.35), "Foundations": (1, 0.35),
        "Frame A": (2, 1.0), "Frame B": (2, 0.1), "Utilities": (2, -0.9),
        "Roof": (3, 0.6), "Fit-out": (3, -0.5),
    }
    G = nx.DiGraph()
    G.add_edges_from(DEPS)
    nx.draw_networkx_edges(G, xy, ax=ax[1], edge_color="0.5", arrowsize=13,
                           node_size=1750, width=1.2)
    nx.draw_networkx_nodes(G, xy, ax=ax[1], nodelist=nodes, node_size=1750,
                           node_color=[PALETTE[cls[pl[n]]] for n in nodes],
                           edgecolors="0.35")
    nx.draw_networkx_labels(G, xy, ax=ax[1], font_size=7.5,
                            labels={n: n.replace("Foundations", "Found.") for n in nodes})
    for i in range(4):
        ax[1].text(i, 1.62, f"$u={i}$", ha="center", fontsize=11, color="0.2")
        ax[1].axvline(i, color="0.9", lw=8, zorder=0)
    ax[1].set_title("(B) Dependency network", loc="left", fontsize=13)
    ax[1].set(xlim=(-0.55, 3.55), ylim=(-1.45, 1.95))
    ax[1].axis("off")

    # --- (C) the two coarse-grainings ---------------------------------------
    a = ax[2]
    dps_nodes = sorted(cls, key=lambda s: (len(s), sorted(s)))
    dps_x = {c: 0.35 + i * 1.05 for i, c in enumerate(dps_nodes)}
    yD, yU = 0.65, -0.75

    for s, t in {(pl[s], pl[t]) for s, t in DEPS}:
        a.annotate("", xy=(dps_x[t], yD), xytext=(dps_x[s], yD),
                   arrowprops=dict(arrowstyle="->", color="0.5", lw=1.1,
                                   connectionstyle="arc3,rad=-0.5",
                                   shrinkA=12, shrinkB=12))
    for c, px in dps_x.items():
        a.scatter([px], [yD], s=430, color=PALETTE[cls[c]], edgecolors="0.35", zorder=3)
    a.text(-0.35, yD, "DPS\nnetwork", ha="right", va="center", fontsize=11)

    m = max(u.values()) + 1
    for s, t in {(u[s], u[t]) for s, t in DEPS}:
        a.annotate("", xy=(0.35 + t * 1.05, yU), xytext=(0.35 + s * 1.05, yU),
                   arrowprops=dict(arrowstyle="->", color="0.5", lw=1.1,
                                   connectionstyle="arc3,rad=-0.5",
                                   shrinkA=12, shrinkB=12))
    for i in range(m):
        a.scatter([0.35 + i * 1.05], [yU], s=430, color="0.78",
                  edgecolors="0.35", zorder=3)
        a.text(0.35 + i * 1.05, yU, str(i), ha="center", va="center", fontsize=10)
    a.text(-0.35, yU, "DUS\nnetwork", ha="right", va="center", fontsize=11)

    a.annotate("", xy=(2.2, yU + 0.42), xytext=(2.2, yD - 0.42),
               arrowprops=dict(arrowstyle="->", color="0.35", lw=1.6))
    a.text(2.35, (yD + yU) / 2, "coarsen", fontsize=10, color="0.3", va="center")
    a.text(2.6, yD + 0.62, "7 activities $\\to$ 5 DPS $\\to$ 4 DUS",
           fontsize=10.5, ha="center", color="0.2")
    a.set_title("(C) Coarse-grained networks", loc="left", fontsize=13)
    a.set(xlim=(-2.0, 5.3), ylim=(-1.65, 1.55))
    a.axis("off")

    fig.subplots_adjust(wspace=0.02)
    savefig(fig, "fig1.pdf")


# ---------------------------------------------------------------------------
# Figure 2 -- DUS profiles of two real schedules (Reviewer 3 point 4)
# ---------------------------------------------------------------------------

def _dus_profile(ax_arc, ax_bar, levels, counts, progress, edges,
                 arc_quantile=0.95):
    """Linear DUS profile: arcs = skip dependencies, bars = activities per level.

    Reviewer 2 found the original semicircular rendering hard to read. Laying the
    DUS spine out on a linear axis, with the skip dependencies in a separate
    panel above, keeps the same information but separates the three things being
    shown: the spine, the size of each level, and the long-range dependencies.
    """
    m = len(levels)
    xlim = (-2, m + 1)

    # --- skip dependencies, drawn as semicircles whose radius is the span ----
    skips = {(i, j): w for (i, j), w in edges.items() if j > i + 1}
    thr = np.quantile(list(skips.values()), arc_quantile)
    shown = {k: w for k, w in skips.items() if w >= thr}
    wmax = max(shown.values())
    theta = np.linspace(0, np.pi, 60)
    for (i, j), w in sorted(shown.items(), key=lambda kv: kv[1]):
        r = (j - i) / 2.0
        f = w / wmax
        ax_arc.plot((i + j) / 2 + r * np.cos(theta), r * np.sin(theta),
                    color=str(max(0.80 - 0.78 * f, 0.03)), lw=0.4 + 1.8 * f,
                    solid_capstyle="round")
    ax_arc.set(xlim=xlim, ylim=(0, m * 0.56))
    ax_arc.axis("off")
    ax_arc.text(0.995, 0.98,
                f"{len(skips):,} dependencies between non-adjacent DUS levels "
                f"(strongest {100 * (1 - arc_quantile):.0f}% drawn)",
                transform=ax_arc.transAxes, ha="right", va="top", fontsize=8,
                color="0.35")

    # --- activities per level, coloured by mean progress ---------------------
    cmap = plt.get_cmap("turbo")
    ax_bar.bar(levels, counts, width=1.0,
               color=cmap(np.asarray(progress) / 100.0), linewidth=0)
    ax_bar.set(yscale="log", xlim=xlim, ylim=(0.8, max(counts) * 2.2))
    ax_bar.set_xlabel("DUS level (position along the DUS spine)")
    ax_bar.set_ylabel("Activities")
    for s in ["top", "right"]:
        ax_bar.spines[s].set_visible(False)


def fig2():
    from dus import DependencyUpriseStructure

    fig, axes = plt.subplots(2, 2, figsize=(13, 4.6), sharex=False,
                             gridspec_kw=dict(height_ratios=[1.0, 1.5], hspace=0.05))
    for j, i in enumerate([1, 2]):
        task = pd.read_parquet(EXAMPLES / f"task_{i}.parquet")
        rel = pd.read_parquet(EXAMPLES / f"task_rel_{i}.parquet")
        d = DependencyUpriseStructure(rel.pred_task_id.values, rel.succ_task_id.values)
        levels, counts = np.unique(d.uprise, return_counts=True)
        prop = "physical_complete_pct"
        agg = d.get_nodes_data_aggregated_by_uprise(
            task[["task_id", prop]], "task_id", {prop: "mean"})
        axes[0, j].set_title(
            f"({'AB'[j]}) Project {i}: {len(task):,} activities, "
            f"{len(rel):,} dependencies, {len(levels)} DUS levels",
            loc="left", fontsize=10.5)
        _dus_profile(axes[0, j], axes[1, j], levels, counts,
                     agg[prop].values, d.uprise_edges)

    fig.colorbar(axes[1, 1].scatter([0, 0], [0, 1], c=[0, 100], cmap="turbo", s=0),
                 ax=axes, orientation="vertical", fraction=0.018, pad=0.02,
                 label="Progress %")
    savefig(fig, "fig2.pdf")


# ---------------------------------------------------------------------------
# scaling figures
# ---------------------------------------------------------------------------

def _scatter_by_company(ax, d, xcol, ycol):
    for c in COMPANIES:
        s = d[d.company == c]
        ax.scatter(s[xcol], s[ycol], color=CB[c], s=12, alpha=0.55,
                   edgecolors="none", label=c.capitalize())


def _ci_band(ax, xs, form, key):
    """Shade the bootstrap 95% band of the selected fit."""
    v = RES[key][form]
    lo, hi = np.array(v["ci"])[:, 0], np.array(v["ci"])[:, 1]
    if form == "log2|multiplicative" or form.startswith("log2"):
        band = [p * np.log(xs) + c for p in (lo[0], hi[0]) for c in (lo[1], hi[1])]
    else:
        band = [a * xs**b for a in (lo[0], hi[0]) for b in (lo[1], hi[1])]
    band = np.array(band)
    ax.fill_between(xs, band.min(axis=0), band.max(axis=0), color="k", alpha=0.13,
                    lw=0, zorder=1)


def _binned_median(ax, x, y, nbins=12):
    """Overlay binned medians with interquartile ranges as a goodness-of-fit aid."""
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)
    idx = np.digitize(x, edges) - 1
    cx, cy, lo, hi = [], [], [], []
    for b in range(nbins):
        s = idx == b
        if s.sum() < 8:
            continue
        cx.append(np.sqrt(edges[b] * edges[b + 1]))
        cy.append(np.median(y[s]))
        lo.append(np.percentile(y[s], 25))
        hi.append(np.percentile(y[s], 75))
    cx, cy = np.array(cx), np.array(cy)
    ax.errorbar(cx, cy, yerr=[cy - np.array(lo), np.array(hi) - cy], fmt="ks",
                ms=5, lw=1.4, capsize=3, zorder=5, label="binned median (IQR)")


def fig3():
    d = load()
    v = RES["dus_scaling"]["log2|multiplicative"]
    p, c = v["params"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set(xlabel="Number of dependencies", ylabel="Number of DUS levels",
           xscale="log")
    _scatter_by_company(ax, d, "n_dependencies", "n_generations")
    xs = np.logspace(np.log10(d.n_dependencies.min()),
                     np.log10(d.n_dependencies.max()), 200)
    _ci_band(ax, xs, "log2|multiplicative", "dus_scaling")
    ax.plot(xs, p * np.log(xs) + c, "k--", lw=2,
            label=f"${p:.1f}\\,\\ln x - {abs(c):.0f}$")
    _binned_median(ax, d.n_dependencies.to_numpy(float),
                   d.n_generations.to_numpy(float))
    ax.set_xlim(d.n_dependencies.min() * 0.7, d.n_dependencies.max() * 1.4)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4, fontsize=10,
              labelspacing=0.3)
    ax.text(0.97, 0.05, f"$n={RES['dus_scaling']['_n']}$\n$R^2={v['r2']:.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
    savefig(fig, "fig3.pdf")


def fig4():
    d = load()
    da = RES["distance"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set(xlabel="Number of DUS levels $m$",
           ylabel="Mean distance to final level",
           xscale="log", yscale="log")
    _scatter_by_company(ax, d, "n_generations", "generation_average_distance_to_end")
    xs = np.logspace(np.log10(d.n_generations.min()),
                     np.log10(d.n_generations.max()), 200)
    ax.plot(xs, (xs - 1) / 2, "k--", lw=2, label="chain, $(m-1)/2$")
    med = da["L_summary"]["median"]
    ax.axhline(med, color="0.25", ls=":", lw=2, label=f"median $= {med:.1f}$")
    _binned_median(ax, d.n_generations.to_numpy(float),
                   d.generation_average_distance_to_end.to_numpy(float), nbins=8)
    ax.set_xlim(d.n_generations.min() * 0.8, d.n_generations.max() * 1.3)
    ax.set_ylim(0.8, 400)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.5, fontsize=10,
              labelspacing=0.3)
    savefig(fig, "fig4.pdf")


def fig5():
    d = load()
    v = RES["dps_scaling"]["power|multiplicative"]
    a, b = v["params"]
    bci = v["ci"][1]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.set(xlabel="Number of dependencies", ylabel="Number of DPS classes",
           xscale="log", yscale="log")
    _scatter_by_company(ax, d, "n_dependencies", "n_fibrations")
    xs = np.logspace(np.log10(d.n_dependencies.min()),
                     np.log10(d.n_dependencies.max()), 200)
    _ci_band(ax, xs, "power|multiplicative", "dps_scaling")
    ax.plot(xs, a * xs**b, "k--", lw=2,
            label=f"${a:.2f}\\,x^{{{b:.2f}}}$")
    ax.plot(xs, xs * (a * xs[0] ** b / xs[0]), color="0.5", ls="-.", lw=1.5,
            label="linear reference")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4, fontsize=11)
    ax.text(0.97, 0.05,
            f"$q={b:.2f}$ [{bci[0]:.2f}, {bci[1]:.2f}]\n$R^2={v['r2']:.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
    savefig(fig, "fig5.pdf")


# ---------------------------------------------------------------------------
# Figure 6 -- PSPLIB replication (Reviewer 1 point 4)
# ---------------------------------------------------------------------------

def fig6():
    """PSPLIB replication, with the construction fits extrapolated out of sample."""
    p = pd.read_csv(HERE / "psplib_instances.csv")
    ex = pd.read_csv(HERE / "public_examples.csv")
    d = load()

    dus_p = RES["dus_scaling"]["log2|multiplicative"]["params"]
    dps_p = RES["dps_scaling"]["power|multiplicative"]["params"]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
    marks = {"j30": "o", "j60": "s", "j90": "^", "j120": "D"}

    for k, (ycol_p, ycol_d, ylab, pred) in enumerate([
        ("n_dus", "n_generations", "Number of DUS levels",
         lambda z: dus_p[0] * np.log(z) + dus_p[1]),
        ("n_dps", "n_fibrations", "Number of DPS classes",
         lambda z: dps_p[0] * z ** dps_p[1]),
    ]):
        a = ax[k]
        a.set(xlabel="Number of dependencies", ylabel=ylab,
              xscale="log", yscale="log")
        a.scatter(d.n_dependencies, d[ycol_d], color="0.82", s=9,
                  edgecolors="none", label="Construction (this study)", zorder=1)
        for s, mk in marks.items():
            q = p[p.dataset == s]
            a.scatter(q.n_dependencies, q[ycol_p], s=13, marker=mk, alpha=0.6,
                      edgecolors="none", label=f"PSPLIB {s}", zorder=2)
        a.scatter(ex.n_dependencies, ex[ycol_p], s=110, marker="*", color="k",
                  zorder=4, label="Public example schedules")

        xs = np.logspace(np.log10(30), np.log10(2e5), 200)
        yy = pred(xs)
        ok = yy > 0
        a.plot(xs[ok], yy[ok], "k--", lw=1.8, zorder=3,
               label="construction fit, extrapolated")

        # out-of-sample agreement on the PSPLIB instances
        ratio = p[ycol_p].to_numpy(float) / np.clip(pred(p.n_dependencies.to_numpy(float)), 1e-9, None)
        a.text(0.97, 0.06,
               f"PSPLIB observed / predicted\nmedian $= {np.median(ratio):.2f}$",
               transform=a.transAxes, ha="right", va="bottom", fontsize=9.5,
               color="0.25")
        a.set_title(f"({'AB'[k]})", loc="left")
        if k == 0:
            a.legend(loc="upper left", frameon=False, fontsize=8.5,
                     handletextpad=0.3, labelspacing=0.22)
    fig.subplots_adjust(wspace=0.25)
    savefig(fig, "fig6.pdf")


def fig7():
    """Small-world test: clustering and distance against degree-preserving nulls."""
    d = pd.read_csv(HERE / "smallworld.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for k, (obs, null, lab) in enumerate([
        ("C", "C_spine", "Transitive clustering $C$"),
        ("L", "L_spine", "Mean distance to final level $L$"),
    ]):
        a = ax[k]
        s = d[np.isfinite(d[obs]) & np.isfinite(d[null]) & (d[null] > 0)]
        a.scatter(s[null], s[obs], s=16, alpha=0.55, color="#377eb8",
                  edgecolors="none")
        lo = min(s[null].min(), s[obs].min())
        hi = max(s[null].max(), s[obs].max())
        pad = 0.05 * (hi - lo)
        a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.6,
               label="observed = null")
        a.set(xlabel=f"{lab}, degree-preserving null", ylabel=f"{lab}, observed",
              xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad))
        r = (s[obs] / s[null]).median()
        a.text(0.04, 0.94, f"({'AB'[k]}) median ratio $= {r:.2f}$",
               transform=a.transAxes, va="top", fontsize=11)
        a.legend(loc="lower right", frameon=False, fontsize=10)
    fig.subplots_adjust(wspace=0.28)
    savefig(fig, "fig7.pdf")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
