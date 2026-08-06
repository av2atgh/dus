"""Statistical analysis backing the revised manuscript.

Addresses Reviewer 1 points 2 (fitting method, goodness of fit, model selection,
confidence intervals, parameter commonality) and 4 (descriptive statistics), and
Reviewer 3 point 5 (the flat distance trend).

Method
------
Candidate functional forms (logarithm with one or two parameters, power law,
linear) are fitted by maximum likelihood under two error families:

  additive        y = f(x) + eps,          eps ~ N(0, s^2)
  multiplicative  y = f(x) * exp(eps),     eps ~ N(0, s^2)

Log-likelihoods for the multiplicative family include the Jacobian term
-sum(log y), so that all eight fits are likelihoods of the *same* observed
quantity y and AIC/BIC are directly comparable across both functional form and
error structure. Confidence intervals are nonparametric case-resampling
bootstraps. Parameter commonality across companies is tested by a likelihood
ratio test of the pooled fit against company-specific fits.

Run:  python analysis/stats.py
Outputs: analysis/results.json, analysis/table1.tex, printed summary.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, optimize

HERE = Path(__file__).parent
REPO = HERE.parent

# Per-project summary statistics of the construction schedules. This file is
# derived from proprietary Nodes & Links data and is not distributed with the
# repository; set DUS_SUMMARY_CSV to point at it. Everything in analysis/psplib.py
# runs on public data and needs no such file.
CSV = Path(os.environ.get("DUS_SUMMARY_CSV",
                          REPO / "data" / "schedule_summary.csv"))

# The three contributing companies are anonymised throughout the manuscript as
# blue / green / orange. The colour of a company is assigned by its number of
# projects in ascending order, so this file carries no client identifiers; the
# assignment is reproduced exactly from the data itself.
COMPANIES = ["blue", "green", "orange"]

RNG = np.random.default_rng(20260806)
NBOOT = 2000

# Excluded from the original analysis: a degenerate schedule whose first DUS
# level connects to every other level, so its DUS network is a star.
EXCLUDE = ["BRDerJ49wW8R62D"]


def load():
    if not CSV.exists():
        raise SystemExit(
            f"Summary CSV not found at {CSV}.\n"
            "This file is derived from proprietary schedule data and is not "
            "distributed with the repository. Set DUS_SUMMARY_CSV to its "
            "location, or run analysis/psplib.py, which uses only public data.")
    d = pd.read_csv(CSV)
    d = d.loc[~d.version_ref.isin(EXCLUDE)].reset_index(drop=True)
    order = d.tenant_ref.value_counts().sort_values().index.tolist()
    d["company"] = d.tenant_ref.map(dict(zip(order, COMPANIES)))
    return d


# ---------------------------------------------------------------------------
# candidate functional forms
# ---------------------------------------------------------------------------

def m_log1(x, p):
    return p * np.log(x)


def m_log2(x, p, c):
    return p * np.log(x) + c


def m_pow(x, a, b):
    return a * x**b


def m_lin(x, a, b):
    return a * x + b


FORMS = {
    "log1": (m_log1, [10.0], 1),
    "log2": (m_log2, [10.0, 0.0], 2),
    "power": (m_pow, [1.0, 0.3], 2),
    "linear": (m_lin, [1e-3, 100.0], 2),
}


# ---------------------------------------------------------------------------
# likelihoods
# ---------------------------------------------------------------------------

def fit(f, x, y, p0, error):
    """Maximum-likelihood fit. Returns (params, loglik of y, residuals)."""
    n = len(x)
    if error == "additive":
        popt, _ = optimize.curve_fit(f, x, y, p0=p0, maxfev=400000)
        resid = y - f(x, *popt)
        s2 = np.mean(resid**2)
        ll = -0.5 * n * (np.log(2 * np.pi * s2) + 1)
    else:  # multiplicative: least squares on log y, plus Jacobian
        def g(xx, *p):
            v = f(xx, *p)
            return np.log(np.clip(v, 1e-12, None))

        popt, _ = optimize.curve_fit(g, x, np.log(y), p0=p0, maxfev=400000)
        resid = np.log(y) - g(x, *popt)
        s2 = np.mean(resid**2)
        ll = -0.5 * n * (np.log(2 * np.pi * s2) + 1) - np.sum(np.log(y))
    return popt, float(ll), resid, float(s2)


def ic(ll, n, k):
    kk = k + 1  # + sigma
    return 2 * kk - 2 * ll, kk * np.log(n) - 2 * ll


def r2(resid, target):
    return float(1.0 - np.sum(resid**2) / np.sum((target - target.mean()) ** 2))


def boot_ci(f, x, y, p0, error, nboot=NBOOT):
    n = len(x)
    out = []
    for _ in range(nboot):
        idx = RNG.integers(0, n, n)
        try:
            popt, *_ = fit(f, x[idx], y[idx], p0, error)
            out.append(popt)
        except Exception:
            continue
    out = np.asarray(out)
    return np.percentile(out, [2.5, 97.5], axis=0).T


def diagnostics(x, resid):
    """Breusch-Pagan heteroscedasticity test and residual normality."""
    z = np.log(x)
    z = (z - z.mean()) / z.std()
    u = resid**2 / np.mean(resid**2)
    slope, intercept = np.polyfit(z, u, 1)
    fitted = slope * z + intercept
    ess = np.sum((fitted - u.mean()) ** 2)
    bp = 0.5 * ess
    return dict(
        bp_stat=float(bp),
        bp_p=float(stats.chi2.sf(bp, 1)),
        shapiro_p=float(stats.shapiro(resid[: min(len(resid), 4500)]).pvalue),
        resid_skew=float(stats.skew(resid)),
    )


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------

def model_selection(x, y, label):
    res = {"_label": label, "_n": int(len(x))}
    for error in ["additive", "multiplicative"]:
        for name, (f, p0, k) in FORMS.items():
            try:
                popt, ll, resid, s2 = fit(f, x, y, p0, error)
            except Exception as e:
                res[f"{name}|{error}"] = dict(error=str(e))
                continue
            a, b = ic(ll, len(x), k)
            target = y if error == "additive" else np.log(y)
            res[f"{name}|{error}"] = dict(
                form=name,
                errmodel=error,
                params=[float(v) for v in popt],
                ci=[[float(lo), float(hi)] for lo, hi in boot_ci(f, x, y, p0, error)],
                r2=r2(resid, target),
                loglik=float(ll),
                aic=float(a),
                bic=float(b),
                sigma=float(np.sqrt(s2)),
                **diagnostics(x, resid),
            )
    keys = [k for k in res if "|" in k and "aic" in res[k]]
    best = min(keys, key=lambda k: res[k]["aic"])
    for k in keys:
        res[k]["daic"] = res[k]["aic"] - res[best]["aic"]
    res["_best"] = best
    return res


def commonality(d, ycol, form, error):
    """Likelihood ratio test: pooled parameters vs company-specific parameters."""
    f, p0, k = FORMS[form]
    x = d.n_dependencies.to_numpy(float)
    y = d[ycol].to_numpy(float)
    n = len(x)
    _, ll_pooled, _, _ = fit(f, x, y, p0, error)

    ll_free = 0.0
    per = {}
    for c in COMPANIES:
        s = d[d.company == c]
        xs = s.n_dependencies.to_numpy(float)
        ys = s[ycol].to_numpy(float)
        popt, ll, resid, _ = fit(f, xs, ys, p0, error)
        ll_free += ll
        per[c] = dict(
            n=int(len(s)),
            params=[float(v) for v in popt],
            ci=[[float(lo), float(hi)] for lo, hi in boot_ci(f, xs, ys, p0, error, 1000)],
            r2=r2(resid, ys if error == "additive" else np.log(ys)),
        )

    df = 2 * (k + 1)  # 3(k+1) free vs (k+1) pooled
    lr = 2 * (ll_free - ll_pooled)
    aic_p, bic_p = ic(ll_pooled, n, k)
    aic_f, bic_f = ic(ll_free, n, 3 * k + 2)
    return dict(
        form=form,
        errmodel=error,
        per_company=per,
        loglik_pooled=float(ll_pooled),
        loglik_free=float(ll_free),
        lr_stat=float(lr),
        df=int(df),
        p_value=float(stats.chi2.sf(lr, df)),
        aic_pooled=float(aic_p),
        aic_free=float(aic_f),
        bic_pooled=float(bic_p),
        bic_free=float(bic_f),
    )


def nested_commonality(d, ycol, kind):
    """Nested test of parameter commonality across companies.

    Three nested models, all fitted by maximum likelihood under multiplicative
    (log-normal) errors:

      pooled        one shared (slope, offset)
      common_slope  shared logarithmic coefficient / power-law exponent,
                    company-specific offset / prefactor
      free          fully company-specific parameters

    This separates the question "is the *form* of the relation shared?" from
    "are the nuisance parameters identical?".
    """
    x = d.n_dependencies.to_numpy(float)
    y = d[ycol].to_numpy(float)
    n = len(y)
    comp = d.company.values
    idx = {c: (comp == c) for c in COMPANIES}

    if kind == "log":
        base = lambda xx, slope, off: slope * np.log(xx) + off
    else:
        base = lambda xx, expo, pre: pre * xx**expo

    def make_nll(mode):
        def nll(par):
            if mode == "pooled":
                pred = base(x, par[0], par[1])
            elif mode == "common_slope":
                pred = np.empty_like(x)
                for i, c in enumerate(COMPANIES):
                    pred[idx[c]] = base(x[idx[c]], par[0], par[1 + i])
            else:
                pred = np.empty_like(x)
                for i, c in enumerate(COMPANIES):
                    pred[idx[c]] = base(x[idx[c]], par[2 * i], par[2 * i + 1])
            r = np.log(y) - np.log(np.clip(pred, 1e-9, None))
            s2 = np.mean(r**2)
            return 0.5 * n * (np.log(2 * np.pi * s2) + 1)
        return nll

    seed = [14.3, -34.0] if kind == "log" else [0.69, 1.56]
    starts = {
        "pooled": seed,
        "common_slope": [seed[0]] + [seed[1]] * 3,
        "free": seed * 3,
    }
    res = {}
    for mode in ["pooled", "common_slope", "free"]:
        nll = make_nll(mode)
        r = optimize.minimize(nll, starts[mode], method="Nelder-Mead",
                              options=dict(maxiter=200000, maxfev=200000,
                                           xatol=1e-8, fatol=1e-8))
        r = optimize.minimize(nll, r.x, method="Nelder-Mead",
                              options=dict(maxiter=200000, maxfev=200000,
                                           xatol=1e-10, fatol=1e-10))
        k = len(starts[mode]) + 1
        res[mode] = dict(loglik=float(-r.fun), k=int(k), aic=float(2 * k + 2 * r.fun),
                         params=[float(v) for v in r.x])
    for a, b in [("pooled", "common_slope"), ("common_slope", "free")]:
        lr = 2 * (res[b]["loglik"] - res[a]["loglik"])
        df = res[b]["k"] - res[a]["k"]
        res[f"lr_{a}_to_{b}"] = dict(stat=float(lr), df=int(df),
                                     p_value=float(stats.chi2.sf(lr, df)))
    res["_best"] = min(["pooled", "common_slope", "free"], key=lambda m: res[m]["aic"])
    res["_kind"] = kind
    return res


def distance_analysis(d):
    m = d.n_generations.to_numpy(float)
    L = d.generation_average_distance_to_end.to_numpy(float)
    res = {"n": int(len(m))}
    rho, p = stats.spearmanr(m, L)
    res["spearman_rho"] = float(rho)
    res["spearman_p"] = float(p)

    for name, f, p0, k in [
        ("constant", lambda z, a: a * np.ones_like(z), [3.0], 1),
        ("log", m_log2, [1.0, 1.0], 2),
        ("power", m_pow, [1.0, 0.2], 2),
        ("linear", m_lin, [0.01, 1.0], 2),
    ]:
        popt, ll, resid, _ = fit(f, m, L, p0, "multiplicative")
        a, b = ic(ll, len(m), k)
        res[name] = dict(
            params=[float(v) for v in popt],
            ci=[[float(lo), float(hi)] for lo, hi in boot_ci(f, m, L, p0, "multiplicative", 1000)],
            r2=r2(resid, np.log(L)),
            aic=float(a),
        )
    best = min(["constant", "log", "power", "linear"], key=lambda k: res[k]["aic"])
    for k in ["constant", "log", "power", "linear"]:
        res[k]["daic"] = res[k]["aic"] - res[best]["aic"]
    res["_best"] = best

    chain = (m - 1) / 2
    ratio = L / chain
    res["ratio_to_chain"] = dict(
        median=float(np.median(ratio)),
        q1=float(np.percentile(ratio, 25)),
        q3=float(np.percentile(ratio, 75)),
        frac_below_1=float(np.mean(ratio < 1)),
    )
    res["L_summary"] = dict(
        median=float(np.median(L)),
        q1=float(np.percentile(L, 25)),
        q3=float(np.percentile(L, 75)),
        p95=float(np.percentile(L, 95)),
        max=float(L.max()),
    )
    for lo, hi, nm in [(0, 50, "small"), (50, 150, "medium"), (150, 10**6, "large")]:
        sel = (m >= lo) & (m < hi)
        res[f"L_by_size_{nm}"] = dict(
            m_range=[float(m[sel].min()), float(m[sel].max())],
            n=int(sel.sum()),
            L_median=float(np.median(L[sel])),
            L_iqr=[float(np.percentile(L[sel], 25)), float(np.percentile(L[sel], 75))],
        )
    return res


def descriptive(d):
    rows = []
    for c in COMPANIES + ["all"]:
        s = d if c == "all" else d[d.company == c]
        r = dict(company=c, n_projects=len(s))
        for key, col in [
            ("act", "n_activities"),
            ("dep", "n_dependencies"),
            ("dus", "n_generations"),
            ("dps", "n_fibrations"),
        ]:
            r[f"{key}_min"] = int(s[col].min())
            # medians of an even-sized sample are half-integers; round rather
            # than truncate so the table is consistent with the quoted values
            r[f"{key}_med"] = int(np.floor(s[col].median() + 0.5))
            r[f"{key}_max"] = int(s[col].max())
        rows.append(r)
    return pd.DataFrame(rows)


def latex_table(t):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrcccc}",
        r"\hline",
        r" & & Activities & Dependencies & DUS levels & DPS classes \\",
        r"Company & Projects & \multicolumn{4}{c}{median [minimum--maximum]} \\",
        r"\hline",
    ]
    for _, r in t.iterrows():
        name = "All" if r["company"] == "all" else r["company"].capitalize()
        if r["company"] == "all":
            lines.append(r"\hline")
        lines.append(
            f"{name} & {r['n_projects']} & "
            f"{r['act_med']:,} [{r['act_min']:,}--{r['act_max']:,}] & "
            f"{r['dep_med']:,} [{r['dep_min']:,}--{r['dep_max']:,}] & "
            f"{r['dus_med']} [{r['dus_min']}--{r['dus_max']}] & "
            f"{r['dps_med']:,} [{r['dps_min']:,}--{r['dps_max']:,}] \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Descriptive statistics of the construction project schedules "
        r"analysed here. The three companies are anonymised and identified by the "
        r"colours used in Figures~\ref{scaling_size}--\ref{scaling_dps}.}",
        r"\label{descriptive}",
        r"\end{table}",
    ]
    return "\n".join(lines).replace(",", "{,}")


def show(res, keys):
    for k in keys:
        v = res[k]
        print(
            f"  {v['form']:7s} {v['errmodel']:14s} R2={v['r2']:6.3f}  dAIC={v['daic']:9.1f}"
            f"  params={np.round(v['params'], 4).tolist()}"
            f"  CI={np.round(v['ci'], 4).tolist()}  BP p={v['bp_p']:.2g}"
        )


def main():
    d = load()
    out = {}

    t = descriptive(d)
    print(t.to_string(index=False))
    (HERE / "table1.tex").write_text(latex_table(t))
    out["descriptive"] = t.to_dict(orient="records")

    x = d.n_dependencies.to_numpy(float)
    keys = [f"{f}|{e}" for e in ["additive", "multiplicative"] for f in FORMS]

    print("\n=== number of DUS levels vs number of dependencies ===")
    ms = model_selection(x, d.n_generations.to_numpy(float), "n_DUS")
    show(ms, keys)
    print("  best:", ms["_best"])
    out["dus_scaling"] = ms

    print("\n=== number of DPS classes vs number of dependencies ===")
    ms2 = model_selection(x, d.n_fibrations.to_numpy(float), "n_DPS")
    show(ms2, keys)
    print("  best:", ms2["_best"])
    out["dps_scaling"] = ms2

    best_dus_form = ms["_best"].split("|")[0]
    best_dus_err = ms["_best"].split("|")[1]
    best_dps_form = ms2["_best"].split("|")[0]
    best_dps_err = ms2["_best"].split("|")[1]

    for tag, ycol, form, err in [
        ("dus_commonality", "n_generations", best_dus_form, best_dus_err),
        ("dps_commonality", "n_fibrations", best_dps_form, best_dps_err),
    ]:
        print(f"\n=== parameter commonality across companies ({tag}, {form}/{err}) ===")
        c = commonality(d, ycol, form, err)
        print(f"  LR = {c['lr_stat']:.1f}, df = {c['df']}, p = {c['p_value']:.3g}")
        for k, v in c["per_company"].items():
            print(f"    {k:7s} n={v['n']:4d} params={np.round(v['params'], 4).tolist()}"
                  f" CI={np.round(v['ci'], 4).tolist()} R2={v['r2']:.3f}")
        print(f"  AIC pooled={c['aic_pooled']:.1f}  free={c['aic_free']:.1f}")
        out[tag] = c

    for tag, ycol, kind in [
        ("dus_nested", "n_generations", "log"),
        ("dps_nested", "n_fibrations", "pow"),
    ]:
        print(f"\n=== nested commonality ({tag}) ===")
        r = nested_commonality(d, ycol, kind)
        for mode in ["pooled", "common_slope", "free"]:
            v = r[mode]
            print(f"  {mode:13s} logL={v['loglik']:9.2f} k={v['k']} AIC={v['aic']:9.1f}"
                  f" params={np.round(v['params'], 4).tolist()}")
        for a, b in [("pooled", "common_slope"), ("common_slope", "free")]:
            v = r[f"lr_{a}_to_{b}"]
            print(f"  LR {a} -> {b}: {v['stat']:.2f}, df={v['df']}, p={v['p_value']:.3g}")
        print("  best AIC:", r["_best"])
        out[tag] = r

    print("\n=== mean distance to end vs number of DUS levels ===")
    da = distance_analysis(d)
    print(f"  n={da['n']}  Spearman rho={da['spearman_rho']:.3f} (p={da['spearman_p']:.3g})")
    for k in ["constant", "log", "power", "linear"]:
        v = da[k]
        print(f"  {k:9s} R2={v['r2']:6.3f} dAIC={v['daic']:8.1f} params={np.round(v['params'], 4).tolist()}"
              f" CI={np.round(v['ci'], 4).tolist()}")
    print("  best:", da["_best"])
    print("  L summary:", {k: round(v, 3) for k, v in da["L_summary"].items()})
    print("  ratio to chain:", {k: round(v, 4) for k, v in da["ratio_to_chain"].items()})
    for nm in ["small", "medium", "large"]:
        print(f"  {nm}:", da[f"L_by_size_{nm}"])
    out["distance"] = da

    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote analysis/results.json and analysis/table1.tex")


if __name__ == "__main__":
    main()
