# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""fluxprune on slow series -> report/prune.html.

    uv run python make_prune_report.py

fluxprune (../fluxprune) keeps a subset of a series' points, read as straight lines between them:
the first, last, minimum and maximum, then the points that matter most for the time-weighted RMSE.
This page measures, on a day of 1 s scans per signal-day cut into 1 h storage blocks
(tslab.prune.signals):

- how close its ranking (bottom-up, then refined) gets to the optimum, against top-down splitting,
  swinging door, LTTB, min/max buckets and even spacing at the same point count;
- how many points each stopping rule keeps where (tslab.prune.rules), each calibrated to the same total;
- whether the rules hold up when the sampling rate or the block length changes;
- what errors should be relative to: the block's range, its largest |value|, the geometric mean of the
  two, or a known span per series, on signals with small variations on a large offset;
- the max_error and max_points guards against swinging door, and peaks() on periodic signals with known peaks;
- what it does to swinging-door archives as input; and how long it takes.
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import html

import _fresh_numba_cache  # noqa: E402, F401  (before numba kernels: a cache keyed on the sources)
import fluxprune
import matplotlib
from fluxprune._guard import errors_kernel, fill_kernel

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from tslab.common.historian import TAG_INFO, TAGS, swinging_door
from tslab.prune import baselines as B
from tslab.prune import rules as R
from tslab.prune.signals import EXTRA_INFO, GUARD, OFFSET, PERIOD_S, PERIODIC, PROMINENCE, SIGNALS, blocks, periodic_scans, scans
from tslab.report.page import clean_outputs, save_fig, shell

OUT = Path(__file__).parent / "report"
PREFIX = "prune-"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
SCANS = "0.72"
DAYS = 5
BUDGET = 36  # points per hour on average: 1% of the 1 s scans
REF_EPS = 0.005  # a reference line on the error curves
PASSES = 2
KNEE_ALPHA = 0.25
QUALITY_KS = (12, 36, 108)
SWEEP_BUDGETS = (12, 36, 108, 324)  # points per hour
OPT_WINDOW, OPT_KS = 600, (12, 24)


def describe(name):
    return TAG_INFO[name][0] if name in TAG_INFO else EXTRA_INFO[name]


# ---------------------------------------------------------------------------
# Rules, with one scale parameter each (calibrated) and the fixed ones bound in
# ---------------------------------------------------------------------------

# name -> (k(r, param, hours), parameter label, lo, hi, description); a parameter that lowers the count as it rises
RULES = {
    "count": (lambda r, K, h, b=BUDGET: R.count(r, K * h), "K", None, None, "K points per hour"),
    "tolerance": (lambda r, eps, h, b=BUDGET: R.rmse(r, eps), "ε", 1e-6, 0.5, "fewest points within RMSE ε"),
    "balance": (lambda r, eps, h, b=BUDGET: R.geomean(r, b * h, eps), "ε", 1e-6, 0.5,
                "√(K·k_ε), K = the budget per hour: the geometric mean of the two"),
    "marginal": (lambda r, lam, h, b=BUDGET: R.marginal(r, lam), "λ", 1e-14, 1e-1, "minimise e² + λk"),
    "marginal (RMSE)": (lambda r, lam, h, b=BUDGET: R.marginal_rmse(r, lam), "λ", 1e-9, 1.0,
                        "minimise e + λk: the least mean RMSE for the total"),
    "amortized": (lambda r, th, h, b=BUDGET: R.amortized(r, th / h), "θ", 1e-9, 1.0,
                  "first k with e / (points added) < θ / hours"),
    "tolerance+noise": (lambda r, eps, h, b=BUDGET: R.rmse_noisy(r, eps), "ε", 1e-6, 0.5,
                        "tolerance, ε raised to the noise σ"),
    "balance+noise": (lambda r, eps, h, b=BUDGET: R.geomean_noisy(r, b * h, eps), "ε", 1e-6, 0.5,
                      "balance, ε raised to the noise σ"),
    "knee": (lambda r, a, h, b=BUDGET: R.knee(r, a), "α", None, None,
             f"first k where 2k points don't improve e by 2^α (α = {KNEE_ALPHA}, not calibrated)"),
}
FIXED = {"count": BUDGET, "knee": KNEE_ALPHA}


def calibrate(rule, ranks, target, budget=BUDGET):
    """The parameter at which the rule keeps `target` points over all blocks (bisection in log space)."""
    k_of, _, lo, hi, _ = RULES[rule]
    if rule == "count":
        return budget
    if rule in FIXED:
        return FIXED[rule]
    lo, hi = np.log(lo), np.log(hi)
    for _ in range(40):
        mid = (lo + hi) / 2
        total = sum(k_of(r, np.exp(mid), 1, budget) for r in ranks)
        lo, hi = (mid, hi) if total > target else (lo, mid)
    return float(np.exp((lo + hi) / 2))


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def corpus(hours=1, days=DAYS, archive=False, step=1, names=SIGNALS):
    out = []
    for name in names:
        for day in range(days):
            for b, (t, x) in enumerate(blocks(name, day, hours, archive)):
                out.append(dict(name=name, day=day, block=b, t=t, x=x, r=fluxprune.rank(t, x)))
    return out


def apply_rules(items, params, hours=1):
    """k and refined RMSE / max error of every rule on every block."""
    res = {rule: {"k": [], "rmse": [], "max": []} for rule in RULES}
    for it in items:
        r = it["r"]
        for rule, (k_of, *_) in RULES.items():
            k = k_of(r, params[rule], hours)
            keep = r.refine(r.keep(k), PASSES)
            res[rule]["k"].append(len(keep))
            res[rule]["rmse"].append(r.error(keep))
            res[rule]["max"].append(B.exact_rmse(it["t"], it["x"], keep, max_err=True)[1])
    return {rule: {key: np.array(v) for key, v in d.items()} for rule, d in res.items()}


def sweep(items):
    """Per budget and calibrated rule: mean and median RMSE (refined), and the most points one block got."""
    out = {}
    for budget in SWEEP_BUDGETS:
        for rule, (k_of, *_) in RULES.items():
            if rule == "knee":
                continue
            p = calibrate(rule, [it["r"] for it in items], budget * len(items), budget)
            ks, es = [], []
            for it in items:
                r = it["r"]
                keep = r.refine(r.keep(k_of(r, p, 1, budget)), PASSES)
                ks.append(len(keep))
                es.append(r.error(keep))
            es = np.array(es)
            out[budget, rule] = {"mean": es.mean(), "median": np.median(es), "maxk": max(ks)}
        print(f"sweep {budget}/h done", flush=True)
    return out


def plot_sweep(sw):
    rules = [rule for rule in RULES if rule != "knee"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, key, title in zip(axes, ("mean", "median", "maxk"),
                              ("mean RMSE / range", "median RMSE / range", "most points in one block")):
        for j, rule in enumerate(rules):
            ax.loglog(SWEEP_BUDGETS, [sw[b, rule][key] for b in SWEEP_BUDGETS], marker="o", ms=4, lw=1.5,
                      color=C[j % len(C)], ls="-" if j < len(C) else "--", label=rule)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("points per hour (corpus average)", fontsize=8)
        ax.set_xticks(SWEEP_BUDGETS, [str(b) for b in SWEEP_BUDGETS])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            (lambda v, _: f"{v:g}") if key == "maxk" else (lambda v, _: f"{100 * v:g}%")))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(labelsize=8)
        tidy(ax, "both")
    axes[0].legend(fontsize=7, frameon=False)
    return save_fig(fig, OUT, f"{PREFIX}sweep", "Mean and median RMSE and the largest block against the budget, per rule")


def quality(items):
    """RMSE and max error of each method at each k, per block (fluxprune refined is the reference)."""
    methods = ["fluxprune", "bottom-up (nested)", "top-down", "swinging door", "LTTB", "min/max buckets", "uniform"]
    out = {k: {m: {"rmse": [], "max": [], "name": []} for m in methods} for k in QUALITY_KS}
    for it in items:
        t, x, r = it["t"], it["x"], it["r"]
        for k in QUALITY_KS:
            picks = {"fluxprune": r.refine(r.keep(k), PASSES), "bottom-up (nested)": r.keep(k),
                     "top-down": B.top_down(t, x, k, r.required), "swinging door": B.sdt(t, x, k),
                     "LTTB": B.lttb(t, x, k), "min/max buckets": B.minmax(t, x, k), "uniform": B.uniform(t, x, k)}
            for m, keep in picks.items():
                e, mx = B.exact_rmse(t, x, keep, max_err=True)
                out[k][m]["rmse"].append(e)
                out[k][m]["max"].append(mx)
                out[k][m]["name"].append(it["name"])
    return out


def optimality(items):
    """On OPT_WINDOW-point windows: each greedy method's RMSE over the optimum's (same required points)."""
    out = {k: {"fluxprune": [], "bottom-up (nested)": [], "top-down": []} for k in OPT_KS}
    for it in items:
        t, x = it["t"][:OPT_WINDOW], it["x"][:OPT_WINDOW]
        if x.max() == x.min():
            continue
        r = fluxprune.rank(t, x)
        for k in OPT_KS:
            best = B.exact_rmse(t, x, B.optimal_with(t, x, k, r.required))
            if best < 1e-9:
                continue
            for m, keep in (("fluxprune", r.refine(r.keep(k), PASSES)), ("bottom-up (nested)", r.keep(k)),
                            ("top-down", B.top_down(t, x, k, r.required))):
                out[k][m].append(B.exact_rmse(t, x, keep) / best)
    return out


def invariance(params):
    """Per rule and signal: median k with 10 s sampling over k with 1 s (same 1 h blocks), and k of a 12 h
    block over the sum of k of its twelve 1 h blocks (day 0)."""
    rate = {rule: {} for rule in RULES}
    length = {rule: {} for rule in RULES}
    for name in SIGNALS:
        hour = blocks(name, 0, 1)
        half = blocks(name, 0, 12)
        for rule, (k_of, *_) in RULES.items():
            p = params[rule]
            ks = [k_of(fluxprune.rank(t, x), p, 1) for t, x in hour]
            ks10 = [k_of(fluxprune.rank(t[::10], x[::10]), p, 1) for t, x in hour]
            rate[rule][name] = float(np.median(np.array(ks10) / np.array(ks)))
            k12 = [k_of(fluxprune.rank(t, x), p, 12) for t, x in half]
            length[rule][name] = float(np.mean([k12[j] / sum(ks[12 * j:12 * j + 12]) for j in range(2)]))
    return rate, length


# what errors are relative to: name -> scale(x finite values, signal) in the values' units
SCALES = {
    "range": lambda f, name: f.max() - f.min(),
    "√(range·max|x|)": lambda f, name: np.sqrt((f.max() - f.min()) * np.abs(f).max()),
    "max|x|": lambda f, name: np.abs(f).max(),
    "span (known)": lambda f, name: SPAN[name],
}
SPAN = {}  # each signal's range over all its days (the furnace's for the furnace with dropouts): a known span
SCALE_RULES = {  # name -> k(r, param, factor = scale / range, hours)
    "tolerance": lambda r, p, f, h: R.rmse(r, p * f),
    "balance": lambda r, p, f, h: R.geomean(r, BUDGET * h, p * f),
    "amortized": lambda r, p, f, h: r.count(error_per_point=p * f / h),
}


def scale_corpus(hours=1):
    """Blocks of the main and the offset signals, with each scale over the block's range."""
    names = SIGNALS + OFFSET
    if not SPAN:
        for name in names:
            x = np.concatenate([scans("furnace" if name == "furnace-dropout" else name, d) for d in range(DAYS)])
            SPAN[name] = float(x.max() - x.min())
    items = corpus(hours, names=names)
    for it in items:
        f = it["x"][np.isfinite(it["x"])]
        it["range"] = float(f.max() - f.min())
        it["f"] = {sc: fn(f, it["name"]) / it["range"] if it["range"] > 0 else 1.0 for sc, fn in SCALES.items()}
    return items


def abs_error(it, keep):
    """RMSE in the values' units; for the furnace with dropouts, over the samples that aren't dropouts (a dropout
    is meant to be kept as a glitch, not traced)."""
    if it["name"] != "furnace-dropout":
        return it["r"].error(keep) * it["range"]
    t = (it["t"] - it["t"][0]).astype(float)
    d = np.interp(t, t[keep], it["x"][keep]) - it["x"]
    return float(np.sqrt(np.mean(d[it["x"] != 0] ** 2)))


def scale_study():
    """Per rule and scale, calibrated to BUDGET per hour over the main and offset signals: points and RMSE over the
    signal's span per block; the least mean RMSE over span possible for the total (e + λk on errors over the span); and
    k of a 12 h block over the sum of k of its twelve 1 h blocks."""
    items = scale_corpus(1)
    target = BUDGET * len(items)
    out = {}
    combos = [(rule, sc) for rule in SCALE_RULES for sc in SCALES] + [("least mean RMSE", "span (known)")]
    for rule, sc in combos:
        k_of = SCALE_RULES.get(rule) or (lambda r, p, f, h: R.marginal_rmse(r, p * f))
        lo, hi = np.log(1e-12), np.log(10.0)
        for _ in range(45):
            mid = (lo + hi) / 2
            total = sum(k_of(it["r"], np.exp(mid), it["f"][sc], 1) for it in items)
            lo, hi = (mid, hi) if total > target else (lo, mid)
        p = float(np.exp((lo + hi) / 2))
        ks, es = [], []
        for it in items:
            r = it["r"]
            keep = r.refine(r.keep(k_of(r, p, it["f"][sc], 1)), PASSES)
            ks.append(len(keep))
            es.append(abs_error(it, keep) / SPAN[it["name"]])
        out[rule, sc] = {"p": p, "k": np.array(ks), "e": np.array(es), "k_of": k_of}
    names = np.array([it["name"] for it in items])
    items12 = scale_corpus(12)
    names12 = np.array([it["name"] for it in items12])
    length = {}
    for (rule, sc), d in out.items():
        if rule not in ("amortized", "least mean RMSE"):
            continue
        k1 = np.array([d["k_of"](it["r"], d["p"], it["f"][sc], 1) for it in items])
        k12 = np.array([d["k_of"](it["r"], d["p"], it["f"][sc], 12) for it in items12])
        length[rule, sc] = {n: k12[names12 == n].sum() / k1[names == n].sum() for n in SIGNALS + OFFSET}
    return items, names, out, length


def plot_scales(items, study):
    """An hour of day 0 of each offset signal (09:00, but the first with a dropout for the furnace with dropouts and the
    first quiet one for the quiet level): the points the amortized rule keeps with errors relative to the range and to
    max|x| (each calibrated to the same total)."""
    def shows(it):
        if it["name"] == "furnace-dropout":
            return bool(np.any(it["x"] == 0))
        return it["range"] < 0.1 if it["name"] == "level-quiet" else it["block"] == 9
    day0 = [it for it in items if it["name"] in OFFSET and it["day"] == 0 and shows(it)]
    picks = [next(it for it in day0 if it["name"] == name) for name in OFFSET]
    fig, axes = plt.subplots(3, 2, figsize=(12, 6.6), layout="constrained")
    for ax, it in zip(axes.T.ravel(), picks):
        t, x, r = it["t"], it["x"], it["r"]
        minutes = (t - t[0]) / 6e10
        ax.plot(minutes, x, color=SCANS, lw=0.6, label="1 s scans")
        title = [it["name"]]
        for j, sc in enumerate(("range", "max|x|")):
            d = study["amortized", sc]
            keep = r.refine(r.keep(d["k_of"](r, d["p"], it["f"][sc], 1)), PASSES)
            ax.plot(minutes[keep], x[keep], "-o", color=C[j], lw=1.2, ms=3, label=f"kept, errors relative to {sc}")
            title.append(f"{sc} {len(keep)}")
        off = ""
        if it["name"] == "furnace-dropout":
            ok = x != 0
            ax.set_ylim(x[ok].min() - 1, x[ok].max() + 1)
            off = f" (dropout to 0 at {minutes[~ok][0]:.0f} min, off the scale)"
        ax.set_title(f"{title[0]}, {int(it['block']):02d}:00: {', '.join(title[1:])} points{off}", loc="left", fontsize=9)
        ax.set_xlim(0, 60)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        tidy(ax)
    for ax in axes[-1]:
        ax.set_xlabel("minutes into the hour", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncols=3, frameon=False, fontsize=9)
    return save_fig(fig, OUT, f"{PREFIX}scales", "One hour of each offset signal with the points the amortized rule keeps "
                    "when errors are relative to the range or to the largest |value|")


CAP = 360  # points per hour: the guard study's safety cap (10% of the scans)


def guard_study():
    """Per block of the main and offset signals (1 h): points, largest error and RMSE of the automatic rule, of
    max_error = the signal's GUARD, of the same with a cap of CAP points, and of swinging door with CompDev = GUARD."""
    items = scale_corpus(1)
    methods = ["automatic", "max_error", f"max_error, max_points {CAP}", "swinging door, CompDev = max_error / 2",
               "swinging door, CompDev = max_error"]
    out = {m: {"k": [], "max": [], "rmse": [], "capped": []} for m in methods}
    for it in items:
        r, x, t = it["r"], it["x"], it["t"]
        tol = GUARD[it["name"]]
        picks = {"automatic": r.prune(), "max_error": r.prune(max_error=tol),
                 f"max_error, max_points {CAP}": r.prune(max_error=tol, max_points=CAP)}
        for m in methods:
            if m.startswith("swinging door"):
                idx = swinging_door((t - t[0]).astype(np.float64), x, tol / 2 if m.endswith("/ 2") else tol, np.inf)
                worst = errors_kernel(r._tau, r._y, idx)[1] * it["range"] if len(r._tau) else 0.0
                capped = False
            else:
                idx, worst, capped = picks[m].index, picks[m].max_error, picks[m].capped
            out[m]["k"].append(len(idx))
            out[m]["max"].append(worst / tol)
            out[m]["rmse"].append(abs_error(it, idx) / SPAN[it["name"]])
            out[m]["capped"].append(capped)
    names = np.array([it["name"] for it in items])
    return names, {m: {key: np.array(v) for key, v in d.items()} for m, d in out.items()}


FILLS = ["swinging door's rule", "farthest reach (fluxprune)", "fewest points (exact)"]


def fill_study():
    """Per block of the main and offset signals: the automatic points (refined) filled at max_error = GUARD by each
    fill: points, RMSE over the span and the fill's time."""
    items = scale_corpus(1)
    out = {m: {"k": [], "rmse": [], "ms": []} for m in FILLS}
    for it in items:
        r = it["r"]
        if not len(r._tau):
            continue
        tol = GUARD[it["name"]] / it["range"]
        base = r.refine(r.keep(r.count()), PASSES)
        calls = {FILLS[0]: lambda: B.other_fill(r._tau, r._y, base, tol, False),
                 FILLS[1]: lambda: fill_kernel(r._tau, r._y, base, tol),
                 FILLS[2]: lambda: B.other_fill(r._tau, r._y, base, tol, True)}
        for m, call in calls.items():
            call()  # compiled and warm
            t0 = time.perf_counter()
            idx = call()
            out[m]["ms"].append(1e3 * (time.perf_counter() - t0))
            out[m]["k"].append(len(idx))
            out[m]["rmse"].append(abs_error(it, idx) / SPAN[it["name"]])
    names = np.array([it["name"] for it in items if len(it["r"]._tau)])
    return names, {m: {key: np.array(v) for key, v in d.items()} for m, d in out.items()}


def fill_section(fills):
    f_names, f = fills
    rows = []
    for name in SIGNALS + OFFSET:
        sel = f_names == name
        rows.append([name] + [f"{d['k'][sel].mean():.0f} · {pct(d['rmse'][sel].mean())}" for d in f.values()])
    rows.append(["<b>all</b>"] + [f"<b>{d['k'].mean():.0f} · {pct(d['rmse'].mean())}</b><br><span class='muted'>"
                                  f"{1e3 * d['ms'].mean():.0f} µs per block, at most {1e3 * d['ms'].max():,.0f} µs</span>"
                                  for d in f.values()])
    return ("<h3>The fill</h3><p>How <code>max_error</code> is met. Each segment of the automatic points that strays "
            "further than max_error is filled from its left end. Every scan after an anchor bounds the slope of a line "
            "from the anchor that passes within max_error of it, from below and above. <i>Swinging door's rule</i> ends "
            "the line at the first scan its chord can't reach. <i>Farthest reach</i>, fluxprune's, scans on while the "
            "bounds still overlap (some line could still pass every scan so far), notes the last scan whose own chord "
            "satisfies them, and when they cross makes that scan the next anchor: with noise, a scan after one the "
            "chord can't reach is often reachable again. <i>Fewest points</i> is the exact minimum (dynamic "
            "programming over every feasible chord). Cells: points per hour · RMSE over the span; the automatic "
            "points the same in each column; below, the fill's time per 1 h block.</p>"
            + table(["signal", *f.keys()], rows))


def windows(t, x, period_ns):
    """The forcing rule: the lowest and highest point of every consecutive window of one period, and the ends."""
    w = (t - t[0]) // period_ns
    edges = np.flatnonzero(np.diff(w)) + 1
    out = {0, len(x) - 1}
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(x)]):
        out |= {a + int(np.argmin(x[a:b])), a + int(np.argmax(x[a:b]))}
    return np.array(sorted(out))


PEAK_METHODS = ["peaks(period)", "peaks(prominence)", "peaks(period, prominence)", "min and max of every period",
                "prune (automatic)", "prune(keep=peaks(period, prominence))"]


def peak_study():
    """Per periodic signal and method, over 1 h blocks of DAYS days: true peaks and troughs kept (those more than half
    a period from a block's ends), kept points that are neither (nor the ends), points, and RMSE over the span."""
    out = {}
    for name in PERIODIC:
        P = np.timedelta64(PERIOD_S[name], "s")
        R = PROMINENCE[name]
        res = {m: {"peaks": [0, 0], "troughs": [0, 0], "extra": 0, "k": 0, "rmse": []} for m in PEAK_METHODS}
        span = np.ptp(np.concatenate([periodic_scans(name, d)[0] for d in range(DAYS)]))
        for day in range(DAYS):
            x_day, tops, bottoms = periodic_scans(name, day)
            for b, (t, x) in enumerate(blocks(name, day, 1)):
                t = t.view("datetime64[ns]")
                lo, hi = 3600 * b, 3600 * (b + 1)
                margin = PERIOD_S[name] // 2
                truth = {"peaks": tops[(tops >= lo + margin) & (tops < hi - margin)] - lo,
                         "troughs": bottoms[(bottoms >= lo + margin) & (bottoms < hi - margin)] - lo}
                every = set((tops[(tops >= lo) & (tops < hi)] - lo).tolist()) | set((bottoms[(bottoms >= lo) & (bottoms < hi)] - lo).tolist())
                pk = fluxprune.peaks(t, x, period=P, prominence=R)
                picks = {"peaks(period)": fluxprune.peaks(t, x, period=P),
                         "peaks(prominence)": fluxprune.peaks(t, x, prominence=R),
                         "peaks(period, prominence)": pk,
                         "min and max of every period": windows(t.view(np.int64), x, PERIOD_S[name] * 1_000_000_000),
                         "prune (automatic)": fluxprune.prune(t, x).index,
                         "prune(keep=peaks(period, prominence))": fluxprune.prune(t, x, keep=pk).index}
                r = fluxprune.rank(t, x)
                for m, idx in picks.items():
                    kept = set(idx.tolist())
                    for kind in ("peaks", "troughs"):
                        # the true extreme, or a scan of the same value within half a period (0.1 °C steps tie)
                        for i in truth[kind]:
                            near = idx[np.searchsorted(idx, i - margin):np.searchsorted(idx, i + margin, "right")]
                            res[m][kind][0] += bool(i in kept or np.any(x[near] == x[i]))
                        res[m][kind][1] += len(truth[kind])
                    res[m]["extra"] += len(kept - every - {0, len(x) - 1})
                    res[m]["k"] += len(idx)
                    res[m]["rmse"].append(r.error(idx) * np.ptp(x) / span)
        out[name] = res
    return out


def timings():
    """Seconds, best of 3, by n (1 s scans, days concatenated): rank; prune with the automatic rule (rank, count,
    refine); prune with max_error = 1% of the range as well; and peaks with a 60 s period and a prominence of 5% of the
    range. A temperature trace and white noise."""
    rows = []
    for name in ("temperature", "white-noise"):
        x_all = np.concatenate([scans(name, d) for d in range(12)])
        for n in (3_600, 43_200, 86_400, 1_000_000):
            x = x_all[:n]
            t = (np.arange(n, dtype=np.int64) * 1_000_000_000).view("datetime64[ns]")
            tol = 0.01 * np.ptp(x)
            calls = {"rank": lambda: fluxprune.rank(t, x),
                     "prune": lambda: fluxprune.prune(t, x),
                     "max_error": lambda: fluxprune.prune(t, x, max_error=tol),
                     "peaks": lambda: fluxprune.peaks(t, x, period=np.timedelta64(60, "s"), prominence=0.05 * np.ptp(x))}
            best = {}
            for key, call in calls.items():
                call()
                best[key] = np.inf
                for _ in range(3):
                    t0 = time.perf_counter()
                    call()
                    best[key] = min(best[key], time.perf_counter() - t0)
            rows.append((name, n, best))
    return rows


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def tidy(ax, grid="y"):
    ax.grid(axis=grid, color="0.9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_examples(items, params, rule):
    """One 1 h block of each signal (day 0, 09:00): the scans and the points the rule keeps."""
    picks = [it for it in items if it["day"] == 0 and it["block"] == 9]
    fig, axes = plt.subplots(len(picks) // 2, 2, figsize=(12, 1.7 * len(picks) // 2), layout="constrained")
    k_of = RULES[rule][0]
    for ax, it in zip(axes.T.ravel(), picks):
        t, x, r = it["t"], it["x"], it["r"]
        keep = r.refine(r.keep(k_of(r, params[rule], 1)), PASSES)
        minutes = (t - t[0]) / 6e10
        ax.plot(minutes, x, color=SCANS, lw=0.6, label="1 s scans")
        ax.plot(minutes[keep], x[keep], "-o", color=C[0], lw=1.2, ms=3, label=f"kept ({rule})")
        ax.set_title(f"{it['name']}: {len(keep)} of {len(x):,} points, RMSE {100 * r.error(keep):.2g}% of range",
                     loc="left", fontsize=9)
        ax.set_xlim(0, 60)
        ax.tick_params(labelsize=8)
        tidy(ax)
    for ax in axes[-1]:
        ax.set_xlabel("minutes into 09:00-10:00", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncols=2, frameon=False, fontsize=9)
    return save_fig(fig, OUT, f"{PREFIX}examples", f"One hour of each signal with the points the {rule} rule keeps")


def plot_curves(items):
    """Relative RMSE against points kept for the 09:00 block of day 0 of each signal, log-log."""
    picks = [it for it in items if it["day"] == 0 and it["block"] == 9]
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), layout="constrained", sharex=True)
    for ax, it in zip(axes.ravel(), picks):
        e = it["r"].rmse()
        k = np.arange(len(e))
        ok = np.isfinite(e) & (e > 0)
        ax.loglog(k[ok], e[ok], color=C[0], lw=1.5)
        ax.axhline(REF_EPS, color=C[1], lw=0.8, ls="--")
        ax.set_ylim(1e-6, 1)
        ax.set_xlim(2, 4000)
        ax.set_title(it["name"], loc="left", fontsize=9)
        ax.tick_params(labelsize=8)
        tidy(ax, "both")
    for ax in axes[-1]:
        ax.set_xlabel("points kept", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE / range", fontsize=8)
    for ax in axes.ravel():
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    return save_fig(fig, OUT, f"{PREFIX}curves", "Error against points kept for one hour of each signal")


def plot_sublinear(res):
    """k of each rule against k of the tolerance rule at the same budget (every block): how far a rule follows the
    complexity the tolerance sees. The slope of a log-log fit: 0 is a fixed count, 1 is the tolerance."""
    c = res["tolerance"]["k"].astype(float)
    shown = ["count", "balance", "balance+noise", "marginal (RMSE)", "amortized", "tolerance+noise"]
    slopes = {rule: np.polyfit(np.log(c), np.log(res[rule]["k"].astype(float)), 1)[0] for rule in RULES if rule != "knee"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained", sharex=True, sharey=True)
    for ax, rule in zip(axes.ravel(), shown):
        ax.loglog(c, res[rule]["k"], "o", ms=2.5, color=C[0], alpha=0.35, mec="none")
        lim = [2, 4000]
        ax.plot(lim, lim, color="0.6", lw=0.8, ls=":")
        ax.set_title(f"{rule}: slope {slopes[rule]:.2f}", loc="left", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        tidy(ax, "both")
    for ax in axes[-1]:
        ax.set_xlabel("points the tolerance rule keeps (same budget)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("points kept", fontsize=8)
    img = save_fig(fig, OUT, f"{PREFIX}sublinear", "Points kept per block against what the tolerance rule keeps, per rule")
    return img, slopes


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def gmean(v):
    v = np.asarray(v, float)
    v = v[v > 0]
    return float(np.exp(np.mean(np.log(v)))) if len(v) else 0.0


def pct(v, digits=2):
    return f"{100 * v:.{digits}g}%"


def table(head, rows):
    """Cells are HTML strings, or (class, HTML) pairs."""
    def td(c):
        return f"<td class='{c[0]}'>{c[1]}</td>" if isinstance(c, tuple) else f"<td>{c}</td>"
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(td(c) for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def shade(ratio):
    """Cell class for an error ratio against fluxprune."""
    return "" if ratio < 1.05 else "r1" if ratio < 1.25 else "r2" if ratio < 1.6 else "r3" if ratio < 2.5 else "r4"


def write(items, params, res, sw, qual, opt, rate, length, scale, guards, fills, peaks_res, archive_res, archive_items,
          times):
    names = SIGNALS
    by = np.array([it["name"] for it in items])
    sections = []
    toc = []

    def section(anchor, title, body):
        toc.append((anchor, title, []))
        sections.append(f"<h2 id='{anchor}'>{html.escape(title)}</h2>\n{body}")

    n_blocks = len(items)
    intro = (f"<p><b>fluxprune</b> keeps a subset of a series' points, read as straight lines between them, for "
             f"storage and processing rather than for compression: the first and last points (to stitch blocks), the "
             f"minimum and maximum, then whichever points lower the <i>time-weighted RMSE</i> of the interpolation most. "
             f"One O(n log n) pass ranks every point (bottom-up: repeatedly drop the point whose removal costs least, exact "
             f"integrals per input segment); a rule picks how many to keep from the error curve that pass records; then "
             f"{PASSES} O(n) refinement passes move each kept point to where it fits best between its neighbours.</p>"
             f"<p class='muted'>Data: {len(names)} synthetic signals × {DAYS} days of 1 s scans, cut into 1 h storage blocks "
             f"({n_blocks:,} blocks of 3,600 points; <code>tslab/prune/signals.py</code>). Errors are relative to each block's "
             f"range (max − min). Regenerate with <code>uv run python make_prune_report.py</code> in "
             f"<code>experimental/</code>.</p>")

    # signals
    rows = [[n, ("l", html.escape(describe(n)))] for n in names]
    section("signals", "The signals", table(["signal", "what it is"], rows)
            + plot_examples(items, params, "amortized"))

    # quality
    body = ("<p>Every method picks the same number of points per 1 h block (day 0, every 4th hour, so 6 blocks per "
            "signal); cells are the geometric mean of the block's RMSE over fluxprune's (refined). fluxprune and "
            "top-down keep the required points within k; swinging door's CompDev is searched for k points (the nearest "
            "count at or above k); LTTB and min/max buckets are the plotting downsamplers.</p>")
    methods = list(qual[QUALITY_KS[0]].keys())
    for k in QUALITY_KS:
        rows = []
        for m in methods:
            cells = [m]
            for name in names:
                d = qual[k][m]
                sel = np.array(d["name"]) == name
                ref = np.array(qual[k]["fluxprune"]["rmse"])[sel]
                mine = np.array(d["rmse"])[sel]
                ok = ref > 1e-12
                if not ok.any():
                    cells.append("<span class='muted'>exact</span>" if np.all(mine < 1e-12) else ("r4", "∞"))
                    continue
                ratio = gmean(mine[ok] / ref[ok])
                cells.append((shade(ratio), f"{ratio:.2f}"))
            all_ref = np.array(qual[k]["fluxprune"]["rmse"])
            mine = np.array(qual[k][m]["rmse"])
            ok = all_ref > 1e-12
            cells.append(f"<b>{gmean(mine[ok] / all_ref[ok]):.2f}</b>")
            mx_ref = np.array(qual[k]["fluxprune"]["max"])
            cells.append(f"{gmean(np.array(qual[k][m]['max'])[ok] / mx_ref[ok]):.2f}")
            rows.append(cells)
        body += f"<h3>{k} points per hour</h3>" + table(["method", *names, "all (RMSE)", "all (max error)"], rows)
    opt_rows = [[m] + [f"{gmean(opt[k][m]):.2f} (worst {max(opt[k][m]):.2f})" for k in OPT_KS] for m in opt[OPT_KS[0]]]
    body += (f"<h3>Against the optimum</h3><p>On the first {OPT_WINDOW} points of the same blocks, the optimal choice of "
             f"k points that include the required ones (dynamic programming, O(n²k)); each greedy method's RMSE over "
             f"the optimum's, geometric mean.</p>"
             + table(["method", *[f"k = {k}" for k in OPT_KS]], opt_rows))
    section("quality", "Ranking quality at a fixed count", body)

    # curves
    section("curves", "Error curves", "<p>What a rule sees: the relative RMSE of keeping k points (the ranking, "
            f"before refinement), one hour of each signal. Smooth signals fall as k⁻² (sine), random walks as k⁻½ "
            f"(temperature, pH), and white noise hardly at all until most points are kept; a step or ramp signal falls "
            f"off a cliff and then sits on its noise. The dashed line is {pct(REF_EPS)} for scale.</p>" + plot_curves(items))

    # rules
    rows = []
    for rule, (_, label, *_rest, desc) in RULES.items():
        d = res[rule]
        p = params[rule]
        cells = [rule, ("l", html.escape(desc)), f"{label} = {p:.3g}"]
        for name in names:
            sel = by == name
            cells.append(f"{d['k'][sel].mean():.0f}<br><span class='muted'>{pct(gmean(d['rmse'][sel]))}</span>")
        cells += [f"{d['k'].sum() / n_blocks:.1f}", pct(np.mean(d["rmse"])), pct(np.median(d["rmse"])),
                  f"{d['k'].max():,}"]
        rows.append(cells)
    img, slopes = plot_sublinear(res)
    body = (f"<p>Each rule's one parameter is set so that all {n_blocks:,} blocks together keep {BUDGET} points per hour on "
            f"average (1% of the scans), except knee, which can't be calibrated (its total is what its α gives). Cells are "
            f"the mean points kept per hour and, below, the geometric mean RMSE of the blocks. The corpus columns are "
            f"the mean points per block, the mean and median block RMSE, and the most points one block got.</p>"
            + table(["rule", "what it does", "parameter", *names, "points/h", "mean", "median", "max k"], rows)
            + "<h3>Sublinear?</h3><p>Each rule's points per block against the tolerance rule's at the same budget "
            "(the tolerance rule follows complexity linearly by definition), every block of the corpus, and the slope "
            "of a log-log fit: 0 is a fixed count, 1 the tolerance, ½ the plain geometric mean. All slopes: "
            + ", ".join(f"{rule} {s:.2f}" for rule, s in slopes.items()) + ".</p>" + img)
    sw_rows = [[rule] + [f"{pct(sw[b, rule]['mean'])} / {pct(sw[b, rule]['median'])}<br><span class='muted'>"
                         f"max k {sw[b, rule]['maxk']:,}</span>" for b in SWEEP_BUDGETS] for rule in RULES if rule != "knee"]
    body += ("<h3>Other budgets</h3><p>The same calibration at other averages: mean / median block RMSE and, below, "
             "the most points one block got.</p>"
             + table(["rule", *[f"{b}/h" for b in SWEEP_BUDGETS]], sw_rows) + plot_sweep(sw))
    rate_rows = [[rule] + [f"{rate[rule][n]:.2f}" for n in names] for rule in RULES]
    length_rows = [[rule] + [f"{length[rule][n]:.2f}" for n in names] for rule in RULES]
    body += ("<h3>Sampling rate and block length</h3><p>A rule that adapts to the data, not to how it was stored, "
             "should keep about as many points of an hour sampled every 10 s as of the same hour sampled every second "
             "(ratio 1 when the signal is captured at both rates; noise can't be), and about as many in one 12 h block "
             "as in twelve 1 h blocks (count and balance scale K with the block's hours). Median over day 0's hours "
             "of k at 10 s over k at 1 s:</p>"
             + table(["rule", *names], rate_rows)
             + "<p>k of a 12 h block over the sum of k of its twelve 1 h blocks (day 0, mean of the two halves):</p>"
             + table(["rule", *names], length_rows))
    section("rules", "How many points: the stopping rules", body)

    # scale
    s_items, s_names, study, slength = scale
    all_names = SIGNALS + OFFSET
    rows = [[n, ("l", html.escape(describe(n)))] for n in OFFSET]
    body = ("<p>Errors so far are relative to each block's range, so a block's precision follows its own range whatever "
            "the units or the offset: an hour of furnace that moves 4 °F around 2,115 °F gets the same relative precision "
            "as an hour of a valve that moves 0-100%. The alternatives are the block's largest absolute value, the "
            "geometric mean of the two, and a known span per series (here each signal's range over all its days, as a "
            "historian's per-tag span would give, but which a store that knows nothing of the series lacks). The ranking "
            "doesn't depend on the scale, only the rules' thresholds do. "
            "Six more signals where it matters:</p>" + table(["signal", "what it is"], rows))
    shown = SIGNALS + OFFSET
    rows = []
    for (rule, sc), d in study.items():
        cells = [rule, sc]
        per = []
        for n in shown:
            sel = s_names == n
            per.append(d["e"][sel].mean())
            cells.append(f"{d['k'][sel].mean():.0f}<br><span class='muted'>{pct(per[-1])}</span>")
        cells += [pct(d["e"].mean()), pct(max(per)), f"{d['k'].max():,}"]
        rows.append(cells)
    body += (f"<p>Each rule and scale calibrated to {BUDGET} points per hour over all {len(s_items):,} blocks of the "
             f"{len(all_names)} signals (5 days each). Cells are the mean points kept per hour and, below, the mean RMSE "
             f"<i>relative to the signal's span</i>: a measure that doesn't depend on the units, the offset or the block, "
             f"so the scales can be compared (the furnace with dropouts is measured off its dropouts). The last row is the "
             f"least mean RMSE over span possible for the total, which needs the span.</p>"
             + table(["rule", "errors relative to", *shown, "mean", "worst signal", "max k"], rows)
             + plot_scales(s_items, study))
    rows = [[rule, sc] + [f"{slength[rule, sc][n]:.2f}" for n in shown]
            + [f"<b>{gmean([slength[rule, sc][n] for n in shown]):.2f}</b>"] for rule, sc in slength]
    body += ("<p>Block length: k of a 12 h block over the sum of k of its twelve 1 h blocks (all days), with θ / hours "
             "for the amortized rule. A ratio below 1 is partly right: twelve blocks keep twelve sets of required points, "
             "and a longer block can spend its points where they matter most.</p>"
             + table(["rule", "errors relative to", *shown, "geometric mean"], rows))
    section("scale", "Relative to what?", body)

    # guards
    g_names, g = guards
    rows = []
    for name in SIGNALS + OFFSET:
        sel = g_names == name
        cells = [name, f"{GUARD[name]:g}"]
        for m, d in g.items():
            cap = f"<br><span class='muted'>capped {100 * d['capped'][sel].mean():.0f}%</span>" if d["capped"][sel].any() else ""
            cells.append(f"{d['k'][sel].mean():.0f} · {d['max'][sel].max():.2g} · {pct(d['rmse'][sel].mean())}{cap}")
        rows.append(cells)
    rows.append(["<b>all</b>", ""] + [f"<b>{d['k'].mean():.0f} · {d['max'].max():.2g} · {pct(d['rmse'].mean())}</b>"
                                      for d in g.values()])
    section("guards", "Guards: max_error and max_points",
            "<p>The automatic rule (θ = 0.0015) alone, then with <code>max_error</code> set to an engineer's value per "
            "signal (the historian tags' CompDev, otherwise about what matters), then with a safety cap of "
            f"<code>max_points</code> = {CAP} per hour (10% of the scans) as well. <code>max_error</code> fills each "
            "segment of the automatic points that strays further with the farthest-reach fill (below). Swinging door at "
            "CompDev = max_error / 2 has the same "
            "guarantee: its doors pivot about the archived point, so at CompDev = max_error (last column) its line "
            "strays up to twice that. Cells: mean points per hour · the largest error of any "
            "block over max_error · mean RMSE over the signal's span; below, the share of blocks the cap bound.</p>"
            + table(["signal", "max_error", *g.keys()], rows)
            + fill_section(fills))

    # peaks
    rows = []
    for name, res in peaks_res.items():
        for m, d in res.items():
            hours = DAYS * 24
            rows.append([name, m, pct(d["peaks"][0] / d["peaks"][1], 3), pct(d["troughs"][0] / d["troughs"][1], 3),
                         f"{d['extra'] / hours:.0f}" if m.startswith(("peaks", "min")) else "",
                         f"{d['k'] / hours:.0f}", pct(np.mean(d["rmse"]))])
    info = "".join(f"<li><b>{n}</b>: {html.escape(describe(n))}; period {PERIOD_S[n]} s, prominence {PROMINENCE[n]:g}</li>"
                   for n in PERIODIC)
    section("peaks", "Peaks and troughs: fluxprune.peaks",
            f"<p>For periodic signals, where each peak and trough matters as a sample of its own: <code>peaks(t, x, "
            f"period=, prominence=)</code> keeps the highest point within half a period on either side, at least "
            f"prominence above its surroundings, and a trough between every two peaks (and the other way round). On "
            f"{DAYS} days of 1 h blocks of:</p><ul>{info}</ul><p>The true peaks and troughs are the highest scan of each "
            f"cycle's upper half (of each object in view) and the lowest of each lower half (each gap); those more than "
            f"half a period from a block's ends are scored, and a kept scan of the same value within half a period "
            f"counts (values in 0.1 °C steps tie). Extra: kept points per hour that are neither (nor a block's "
            f"ends). <i>min and max of every period</i> is the forcing rule: the lowest and highest point of each "
            f"consecutive period.</p>"
            + table(["signal", "method", "peaks kept", "troughs kept", "extra/h", "points/h", "RMSE / span"], rows))

    # archives
    rows = []
    for name in TAGS:
        sel = np.array([it["name"] == name for it in archive_items])
        n_in = sum(len(it["x"]) for it, s in zip(archive_items, sel) if s)
        cells = [name, f"{n_in / (DAYS * 24):.0f}"]
        for rule in ("count", "tolerance", "balance", "tolerance+noise", "balance+noise"):
            k = archive_res[rule]["k"][sel]
            cells.append(f"{k.sum() / (DAYS * 24):.0f} ({pct(k.sum() / n_in)})<br>"
                         f"<span class='muted'>{pct(gmean(archive_res[rule]['rmse'][sel]))}</span>")
        rows.append(cells)
    section("archives", "Swinging-door archives as input",
            "<p>The same rules with the same parameters on what each tag's swinging-door historian archived "
            "(<code>tslab/common/historian.py</code>: exception + swinging-door filters at the tag's CompDev), in 1 h "
            "blocks: points per hour in and kept (share of the archive), and below, the geometric-mean RMSE against "
            "the archive's own straight lines (relative to the block's range).</p>"
            + table(["tag", "archived/h", "count", "tolerance", "balance", "tolerance+noise", "balance+noise"], rows))

    # speed
    rows = []
    for name, n, b in times:
        tr = b["rank"]
        rows.append([name, f"{n:,}", f"{1e3 * tr:.1f} ms", f"{1e9 * tr / n:.0f} ns"]
                    + [f"+{1e3 * (b[key] - tr):.1f} ms (+{100 * (b[key] - tr) / tr:.0f}%)" for key in ("prune", "max_error")]
                    + [f"{1e3 * b['peaks']:.1f} ms ({100 * b['peaks'] / tr:.0f}% of rank)"])
    power = ("" if "AC Power" in subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True).stdout
             else "<b>Measured on battery power: indicative only, rerun on AC.</b> ") if sys.platform == "darwin" else ""
    section("speed", "Speed", f"<p>{power}Apple M3, one thread, best of 3, 1 s scans. rank is the O(n log n) pass; "
            "prune adds picking k (O(n)) and the two O(n) refinement passes, shown as the time it adds to rank; "
            "max_error (1% of the range) adds the swinging-door fill on top of that, also against rank; peaks (60 s period, "
            "prominence 5% of the range) is separate and O(n). <code>bench/prune.py</code> has the same per call "
            "for more signals.</p>"
            + table(["signal", "n", "rank", "per point", "prune (automatic)", "prune, max_error", "peaks"], rows))

    section("findings", "Findings", FINDINGS)
    (OUT / "prune.html").write_text(shell("fluxprune: keeping the points that matter", intro, toc, "\n".join(sections)))


FINDINGS = """
<ol>
<li><b>Ranking: bottom-up, then refine.</b> Removing the cheapest point repeatedly (exact time-weighted L2, a heap,
O(n log n)) gives a nested ranking of every point and the whole error curve in one pass, but greedy merging is myopic:
25-35% above the optimum. Two O(n) passes that slide each kept point to its best spot between its neighbours bring
it to about 6% above the optimum on 600-point windows (worst 1.3×), for +5-15% of the ranking's time. Top-down
splitting is better than plain bottom-up on step-like signals and much worse on others (worst 7× the optimum).
At the same number of points, swinging door's RMSE is 1.3-1.5× fluxprune's, LTTB's 2-2.5×, and min/max buckets' and
even spacing's 2-4×: the plotting downsamplers keep shape for the eye, not the interpolation.</li>
<li><b>The error curve tells the signal apart from the noise.</b> Smooth signals fall as k⁻², random walks as k⁻½,
noise stays flat until most points are kept. The curve at k = n/2 measures the white noise: it is 0.29σ for white
noise at every block size (<code>tslab/prune/rules.py</code>, <code>noise</code>), regular or irregular times.</li>
<li><b>Fixed count</b> spends the same on a valve that two points describe exactly as on white noise, and <b>fixed
tolerance</b> is the opposite: calibrated to the same total it hands noise blocks everything (white noise 194 points/h
against 36 average, a block up to 432) and its ε ends at 13% of the range for all the others: the worst mean
error of all rules. Neither alone is a rate control for mixed data.</li>
<li><b>The geometric mean of the two</b> (balance, √(K·k_ε)) is a real compromise (slope ~0.6 against the tolerance's
counts) and bounded by √(K·n), but noise is still the most "complex" thing it sees: white noise gets 4× the average.
<b>Raising ε to the block's own noise level</b> (balance+noise) fixes that and reaches nearly the least mean RMSE
possible for the total at 36/h (3.4% vs 3.3%) with the smallest largest block of the adaptive rules (132); at large
budgets it stops early on purpose (it won't trace noise), so its mean levels off. Its weakness is the noise estimate:
correlated noise (flow: τ 5 s at 1 s scans) partly reads as signal, so the count depends on the sampling rate there (×0.25).</li>
<li><b>The marginal rules</b> (stop when a point gains less than λ) give the least mean error for the total by
construction (e + λk) but depend on the sampling rate: a noise point's gain scales as 1/n, so the same λ keeps
13-25× more noise points at 10 s than at 1 s. The user's prediction holds: on noise they either give up at once or
never.</li>
<li><b>The amortized rule</b> (stop at the first k where the error left is under θ per point added) is the best
one-knob rule here: within 4-21% of the least mean RMSE at every budget, largest block 2.5-3.7× the average, no noise
estimate, and the most stable under a change of sampling rate (×0.80-0.98 for every signal). It stops where the error
curve meets the line θ·k: in effect error and points traded at a fixed rate, so it grows with complexity
sublinearly (k ∝ c^⅓ for a smooth signal's k⁻² curve, ∝ σ for noise). It is <code>error_per_point</code> in the API.</li>
<li><b>Knee detection</b> (stop when doubling the points stops paying) is unusable: periodic signals have plateaus at
small k (12 h of a sine: 4 points) and random walks never stop (temperature: 3,061 of 3,600).</li>
<li><b>Block length.</b> Every rule but the fixed count keeps fewer points in one 12 h block than in twelve 1 h blocks,
for two reasons. A parameter per block is spread over a longer block: θ / hours for the amortized rule (as K · hours for
a count) takes its geometric mean over the 18 signals from 0.09 to 0.30 (θ per block: 0.01-0.48; θ / hours: 0.01-0.93,
the low end the quiet level and the dropouts, whose range a single event sets). And a longer block has a
wider range: with a known span per series instead, the same rule reaches 0.70. The rest is as it should be: twelve
blocks keep twelve sets of required points, and one block can place its points where they matter most.</li>
<li><b>Relative to what?</b> Of the scales that need no knowledge of the series, the block's own range is the best:
with the amortized rule at 36/h, the mean error over each signal's span is 1.3%, against 1.1% with known spans and 1.0%
the least possible. It gives small variations on a large offset the precision they need by itself: the furnace (4 °F
an hour around 2,115) and the grid frequency keep 27 and 57 points per hour, where known spans give 16 and 62.
<b>The largest |value|</b> depends on where zero is: the same ambient temperature keeps 38 points per hour in °C and 5
in kelvin, and it starves exactly the signals that have little range for their offset (furnace and grid frequency: 4
per hour, at 2.8% and 9% of their span, against 0.34% and 2%). <b>√(range · max|x|)</b> halves both effects but keeps
them (41 against 12 points for °C and K, grid 4% of span), and it helps block length less than a known span (0.43
against 0.70). The range has two costs: a quiet block makes its own dither look like signal (the quiet level keeps 74
points per hour tracing one 0.01 step, against 5 with its span), and a glitch that sets the range takes its block's
precision (furnace with dropouts: 1.1% of span against 0.34%). A known span fixes both, and so does an
absolute <code>max_error</code> guard (next).</li>
<li><b>Swinging-door archives as input</b> prune further, depending on what the archive kept: a tag whose CompDev is
below its noise (flow, motor current: 1,300-3,000 archived points per hour) drops to 0.6-6% of the archive at the
same settings, a smooth tag (temperature, pH: ~330/h) to 2-11%, and an archive that already has two points per hour
stays as it is.</li>
<li><b>max_error</b> only adds points: each segment of the automatic choice that strays further than max_error from the
scans is filled with the farthest-reach fill: from each anchor, the line to the farthest scan it can reach while every
scan between stays within max_error. Swinging door's rule (the line ends at the first scan the chord can't reach) keeps
1.1-3× as many points on the noisy signals (vibration 170 against 66 per hour, batch 72 against 23, temperature 766
against 510), the exact minimum 4% fewer overall at about 35× the time (12 against 440 µs per block, up to 8 ms). At the same guarantee as
swinging door (CompDev = max_error / 2, since at CompDev = max_error its line strays up to 1.6-2× that), fluxprune keeps
fewer points on 12 of the 18 signals (batch 23 against 228 per hour, vibration 66 against 422, temperature 510 against
1,136, grid frequency 282 against 597) and as many on two; more only where the automatic rule already wanted more than
the guard needs (the quiet level's dither: 77 against 2). A max_error below a signal's noise keeps most of the scans,
as swinging door does (motor current: 3,160 per hour); <b>max_points</b> (here 360 per hour) caps that. The budget goes
in a fixed order: the required points; then max_error, if it can be met within the cap, with the automatic points
taking what is left (the least significant give way first, by a bisection on their number); and if it can't,
max_error relaxed to about the tightest bound the cap allows (a bisection over the fill), with no automatic points.
The fill always comes last, since adding a point can make the error worse. The cap bound only on the five
noise-dominated signals; in their worst block the bound reached is 1.2× max_error for temperature, 1.3× for pH, 2.1×
for white noise, 2.7× for flow and 11× for motor current, whose noise is five times max_error. Measured and not
adopted: splitting the worst segment at its worst point when the cap binds (1.3-1.7× larger errors: it anchors on
noise spikes); filling first and ranking again with the fill fixed (the same RMSE at the same number of points, 3-5×
slower); a secant search for the relaxed bound (9.4 fills on average against 12.8, but up to 28 against 14); trying
every number of automatic points instead of bisecting (0.4% lower RMSE on average).</li>
<li><b>Peaks are a different job</b>, so they have their own function. With a 60 s period and a prominence of about
6σ of the noise or more, <code>peaks()</code> keeps every true peak and trough of a vibration whose period drifts from 40
to 80 s and whose amplitude drifts fivefold, and of a pyrometer's objects and the gaps between them, with 0.3 and 1.4
other points per hour. The prominence does most of the work: alone it leaves 0.3 and 10 extra points per hour (noise on
the objects' tops); the period alone leaves 69-78 (every noise bump half a period from a higher one) and misses 3% of
the pyrometer's troughs. Three choices settled on the way: the period compares peaks with peaks (scipy's distance), not
with every scan within half a period, which dropped the peak of a cooler object that follows a hotter one closely (its
tail is still hot: 0.3% of the pyrometer's peaks); the prominence test comes first, so that noise can't suppress a real
peak; and a side of a peak with no higher point before the block's end doesn't count, since the block's edge cuts it
(by the usual definition, the first gap of a block had the prominence of its noise). Forcing a minimum and a maximum
into every period (the first idea) misses 15% of the extremes: real peaks fall on window edges, and windows without one
get a meaningless pair. The automatic rule keeps only a third of the vibration's peaks and a seventh of the
pyrometer's: they cost little RMSE. <code>prune(keep=peaks(...))</code> keeps them all and prunes the rest as usual,
with a lower RMSE than either alone.</li>
</ol>
<p><b>The API.</b> <code>prune(t, x, error_per_point=θ / H)</code> for blocks of H hours is the default (θ = 0.0015
keeps ~1% of 1 s scans on this corpus; the error relative to each block's range). Two optional guards bound it:
<code>max_points</code>, a hard cap, and <code>max_error</code>, the largest error of any scan in value units, which
also takes care of quiet blocks, glitches and the block length for a configured signal. For periodic signals,
<code>peaks(t, x, period=, prominence=)</code>, passed to <code>prune(keep=)</code>. The other rules on this page
(count, tolerance, balance, noise floor, marginal, knee) are experiments in <code>tslab/prune/rules.py</code>.</p>
"""


def main():
    clean_outputs(OUT, [f"{PREFIX}*.svg", "prune.html"])
    t0 = time.time()
    items = corpus()
    print(f"corpus: {len(items)} blocks, {time.time() - t0:.0f} s", flush=True)
    params = {rule: calibrate(rule, [it["r"] for it in items], BUDGET * len(items)) for rule in RULES}
    print("params:", params, flush=True)
    res = apply_rules(items, params)
    sw = sweep(items)
    qual = quality([it for it in items if it["day"] == 0 and it["block"] % 4 == 1])
    print(f"quality: {time.time() - t0:.0f} s", flush=True)
    opt = optimality([it for it in items if it["day"] == 0 and it["block"] % 4 == 1])
    print(f"optimum: {time.time() - t0:.0f} s", flush=True)
    rate, length = invariance(params)
    scale = scale_study()
    print(f"scale: {time.time() - t0:.0f} s", flush=True)
    guards = guard_study()
    fills = fill_study()
    peaks_res = peak_study()
    print(f"guards and peaks: {time.time() - t0:.0f} s", flush=True)
    archive_items = corpus(archive=True, names=TAGS)
    archive_res = apply_rules(archive_items, params)
    times = timings()
    print(f"done: {time.time() - t0:.0f} s", flush=True)
    write(items, params, res, sw, qual, opt, rate, length, scale, guards, fills, peaks_res, archive_res, archive_items,
          times)


if __name__ == "__main__":
    main()
