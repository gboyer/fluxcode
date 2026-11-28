# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""fluxprune on slow series -> report/prune.html.

    uv run python make_prune_report.py

fluxprune (../fluxprune) keeps a subset of a series' points, read as straight lines between them:
the first, last, minimum and maximum, then the points that matter most for the time-weighted RMSE.
This page shows, on a day of 1 s scans per signal-day cut into 1 h storage blocks
(tslab.prune.signals):

- first, how to tune it: the default, a cap and a coarse exchange rate (error_per_point), then a loose
  and a fine max_error with and without a cap, on one hour of eight signals and on all blocks of all;
then, how it works and how it was chosen:
- how close its ranking (bottom-up, then refined) gets to the optimum, against top-down splitting,
  Visvalingam-Whyatt, swinging door, LTTB, min/max buckets and even spacing at the same point count;
- how many points each stopping rule keeps where (tslab.prune.rules), each calibrated to the same total;
- whether the rules hold up when the sampling rate or the block length changes;
- what errors should be relative to: the block's range, its largest |value|, the geometric mean of the
  two, or a known span per series, on signals with small variations on a large offset;
- the noise floor: the rmse pass reading its error above the block's own white noise, on a pyrometer's
  narrow spikes and quiet hours too;
- the max_error and max_points guards against swinging door, and the max_error pass against its alternatives;
- refit=True: least-squares values at the kept times instead of the samples;
- what it does to swinging-door archives as input; and how long it takes.
"""

import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import html

import _fresh_numba_cache  # noqa: E402, F401  (before numba kernels: a cache keyed on the sources)
import fluxprune
import matplotlib
from fluxprune._api import rank
from fluxprune._guard import max_error_kernel

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from tslab.common.historian import TAG_INFO, TAGS, swinging_door
from tslab.prune import baselines as B
from tslab.prune import rules as R
from tslab.prune.signals import EXTRA_INFO, GUARD, MAX_ERROR, NOISY, OFFSET, SIGNALS, blocks, pyrometer, scans
from tslab.report.page import clean_outputs, collapsible_table, save_fig, shell

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
                  "first k with √(e² − σ²) / (points added) < θ / hours, σ the block's noise (the default)"),
    "amortized, no floor": (lambda r, th, h, b=BUDGET: R.amortized(r, th / h, False), "θ", 1e-9, 1.0,
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
                out.append(dict(name=name, day=day, block=b, t=t, x=x, r=rank(t, x)))
    return out


def apply_rules(items, params, hours=1):
    """k and refined RMSE / max error of every rule on every block."""
    res = {rule: {"k": [], "rmse": [], "max": []} for rule in RULES}
    for it in items:
        r = it["r"]
        for rule, (k_of, *_) in RULES.items():
            k = k_of(r, params[rule], hours)
            keep = r.refine(r.top(k), PASSES)
            res[rule]["k"].append(len(keep))
            res[rule]["rmse"].append(r.relative_error(keep))
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
                keep = r.refine(r.top(k_of(r, p, 1, budget)), PASSES)
                ks.append(len(keep))
                es.append(r.relative_error(keep))
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
    methods = ["fluxprune", "bottom-up (nested)", "top-down", "Visvalingam-Whyatt", "Visvalingam-Whyatt + refine",
               "swinging door", "LTTB", "min/max buckets", "uniform"]
    out = {k: {m: {"rmse": [], "max": [], "name": []} for m in methods} for k in QUALITY_KS}
    for it in items:
        t, x, r = it["t"], it["x"], it["r"]
        for k in QUALITY_KS:
            vw = B.visvalingam_whyatt(t, x, k, r.required)
            picks = {"fluxprune": r.refine(r.top(k), PASSES), "bottom-up (nested)": r.top(k),
                     "top-down": B.top_down(t, x, k, r.required), "Visvalingam-Whyatt": vw,
                     "Visvalingam-Whyatt + refine": r.refine(vw, PASSES), "swinging door": B.sdt(t, x, k),
                     "LTTB": B.lttb(t, x, k), "min/max buckets": B.minmax(t, x, k), "uniform": B.uniform(t, x, k)}
            for m, keep in picks.items():
                e, mx = B.exact_rmse(t, x, keep, max_err=True)
                out[k][m]["rmse"].append(e)
                out[k][m]["max"].append(mx)
                out[k][m]["name"].append(it["name"])
    return out


def optimality(items):
    """On whole 1 h blocks: each greedy method's RMSE over the optimum's (same required points)."""
    out = {k: {"fluxprune": [], "bottom-up (nested)": [], "top-down": []} for k in QUALITY_KS}
    for it in items:
        t, x, r = it["t"], it["x"], it["r"]
        if x.max() == x.min():
            continue
        for k in QUALITY_KS:
            best = B.exact_rmse(t, x, B.optimal_with(t, x, k, r.required))
            if best < 1e-9:
                continue
            for m, keep in (("fluxprune", r.refine(r.top(k), PASSES)), ("bottom-up (nested)", r.top(k)),
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
            ks = [k_of(rank(t, x), p, 1) for t, x in hour]
            ks10 = [k_of(rank(t[::10], x[::10]), p, 1) for t, x in hour]
            rate[rule][name] = float(np.median(np.array(ks10) / np.array(ks)))
            k12 = [k_of(rank(t, x), p, 12) for t, x in half]
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
        return it["r"].relative_error(keep) * it["range"]
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
            keep = r.refine(r.top(k_of(r, p, it["f"][sc], 1)), PASSES)
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
            keep = r.refine(r.top(d["k_of"](r, d["p"], it["f"][sc], 1)), PASSES)
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


NOISE_THETAS = (1.5e-3, 1.5e-4, 1.5e-5, 1.5e-6)
NOISE_SHOWN = ["temperature", "flow", "motor-current", "vibration", "white-noise", "sine", "level-quiet", "pyrometer"]
SPIKE_AT = 6.0  # °C above the pyrometer's floor: a scan on a spike (3σ)


def noise_study():
    """The rmse pass with and without the noise floor: per signal at the default θ (all days), the points per
    hour over θ (day 0), and the pyrometer's spikes and quiet hours against its noise-free curve."""
    names = SIGNALS + OFFSET + NOISY
    per = {}
    for name in names:
        d = per[name] = {key: [] for key in ("on", "off", "e_on", "e_off", "don", "doff", "e_don", "e_doff", "sigma")}
        for day in range(DAYS):
            for t, x in blocks(name, day, 1):
                r = rank(t, x)
                for key, theta, floor in (("on", 0.0015, True), ("off", 0.0015, False), ("don", None, True),
                                          ("doff", None, False)):
                    keep = r.refine(r.top(r.count(theta, noise_floor=floor)), PASSES)
                    d[key].append(len(keep))
                    d["e_" + key].append(r.relative_error(keep))
                d["sigma"].append(r._noise)
    archive = {}  # the tags' swinging-door archives: σ, points and RMSE against the scans, floor and none
    for tag in ("temperature", "flow", "ph-float32", "motor-current"):  # the others archive 2-5 points per hour
        d = archive[tag] = {"n": [], "sigma": [], "k": [], "e": [], "k_off": [], "e_off": [], "e_in": []}
        for day in range(DAYS):
            for (ts, xs), (ta, xa) in zip(blocks(tag, day, 1), blocks(tag, day, 1, archive=True)):
                span = float(np.ptp(xs))
                if span <= 0 or len(xa) < 3:
                    continue
                tsf, taf = (ts - ts[0]) / 1e9, (ta - ts[0]) / 1e9
                d["n"].append(len(xa))
                d["e_in"].append(np.sqrt(np.mean((np.interp(tsf, taf, xa) - xs) ** 2)) / span)
                for floor, kk, ee in ((True, "k", "e"), (False, "k_off", "e_off")):
                    p = fluxprune.prune(taf, xa, noise_floor=floor)
                    if floor:
                        d["sigma"].append(p.noise)
                    d[kk].append(len(p))
                    d[ee].append(np.sqrt(np.mean((np.interp(tsf, p.times, p.values) - xs) ** 2)) / span)
    per["archives"] = archive
    zero = {True: [], False: []}  # white noise (σ 1): the kept lines' RMSE against the zero under it
    for day in range(DAYS):
        for t, x in blocks("white-noise", day, 1):
            for floor in zero:
                p = fluxprune.prune(t, x, noise_floor=floor)
                zero[floor].append(float(np.sqrt(np.mean(np.interp(t, p.times, p.values) ** 2))))
    per["white-noise"]["zero"] = zero
    sweep_k = {}
    for name in names:
        rs = [rank(t, x) for t, x in blocks(name, 0, 1)]
        for floor in (True, False):
            sweep_k[name, floor] = [np.mean([r.count(th, floor) for r in rs]) for th in NOISE_THETAS]
    settings = {
        "θ = 0.0015, no floor": dict(error_per_point=0.0015, noise_floor=False),
        "θ = 0.0015": dict(error_per_point=0.0015),
        "default (θ = 2.8e-6 at 1 h)": {},
        "default, no floor": dict(noise_floor=False),
        f"θ = 0.0015, max_error = {GUARD['pyrometer']:g}": dict(error_per_point=0.0015, max_error=GUARD["pyrometer"]),
    }
    return per, sweep_k, pyrometer_study(settings)


def pyrometer_study(settings):
    """Per setting (label -> prune kwargs), over the pyrometer's 1 h blocks of all days against its noise-free
    curve: points in busy and quiet hours, points on spikes, each spike's peak miss, and the RMSE under the noise."""
    pyro = {k: {"busy": [], "quiet": [], "on_spike": [], "miss": [], "under": [], "under_quiet": []} for k in settings}
    for day in range(DAYS):
        x, truth, centers = pyrometer(day)
        s = np.arange(len(x), dtype=np.float64)
        for b in range(24):
            sl = slice(3600 * b, 3600 * (b + 1))
            t, xb, tb = s[sl], x[sl], truth[sl]
            cb = centers[(centers >= t[0] + 5) & (centers <= t[-1] - 5)]
            spike = tb - tb.min() > SPIKE_AT
            for label, kw in settings.items():
                p = fluxprune.prune(t, xb, **kw)
                line = np.interp(t, p.times, p.values)
                d = pyro[label]
                kept = np.isin(t, p.times)
                d["busy" if len(cb) else "quiet"].append(len(p))
                d["on_spike"].append(int(np.count_nonzero(kept & spike)))
                d["under" if len(cb) else "under_quiet"].append(float(np.sqrt(np.mean((line - tb) ** 2))))
                for c in cb:
                    near = np.abs(t - c) <= 5
                    d["miss"].append(float(tb[near].max() - line[near].max()))
    return pyro


def pyrometer_table(pyro):
    rows = []
    for label, d in pyro.items():
        miss = np.abs(np.array(d["miss"]))
        rows.append([label, f"{np.mean(d['busy']):.0f}", f"{np.mean(d['quiet']):.0f}",
                     f"{np.mean(d['on_spike']):.0f}", f"{np.mean(miss):.1f}", f"{np.max(miss):.0f}",
                     f"{np.mean(d['under']):.1f}", f"{np.mean(d['under_quiet']):.2f}"])
    return table(["setting", "points/h, busy", "points/h, quiet", "on spikes/h", "peak miss", "worst",
                  "RMSE, busy", "RMSE, quiet"], rows)


# ---------------------------------------------------------------------------
# Tuning: the settings one would try, on every signal
# ---------------------------------------------------------------------------

CAP = 360  # points per hour: a tenth of the 1 s scans
COARSE = 1.4e-3  # θ: half the range at 360 points (about the first default)
TUNING = {  # figure -> {label: prune kwargs}; max_error "loose" or "fine" is the signal's from MAX_ERROR
    "theta": {
        "default": {},
        f"default, max_points {CAP}": dict(max_points=CAP),
        "coarse": dict(error_per_point=COARSE),
    },
    "max-error": {
        "coarse": dict(error_per_point=COARSE),
        "coarse, loose max_error": dict(error_per_point=COARSE, max_error="loose"),
        "coarse, fine max_error": dict(error_per_point=COARSE, max_error="fine"),
        f"coarse, fine max_error, max_points {CAP}": dict(error_per_point=COARSE, max_error="fine", max_points=CAP),
    },
}
EXAMPLES = ["temperature", "flow", "motor-current", "compressor", "batch", "vibration", "furnace", "pyrometer"]
WHITE_SIGMAS = (1, 2, 3, 5)  # max_error over σ on white noise


def tuning_kwargs(kw, sig):
    """prune's keyword arguments for a signal: a max_error of "loose" or "fine" becomes the signal's value."""
    if isinstance(kw.get("max_error"), str):
        return {**kw, "max_error": MAX_ERROR[sig][kw["max_error"] == "fine"]}
    return kw


def tuning_study():
    """Every tuning setting on every signal's 1 h blocks (all days): points, whether the cap bound (the rmse pass cut
    short or max_error relaxed), the RMSE relative to the block's range and the largest error in the signal's units;
    the noise σ; white noise's points per hour at max_error = a few σ; and the pyrometer against its noise-free curve."""
    every = {label: kw for group in TUNING.values() for label, kw in group.items()}
    out = {label: {} for label in every}
    sigma = {}
    for sig in SIGNALS + OFFSET + NOISY:
        for day in range(DAYS):
            for t, x in blocks(sig, day, 1):
                f = x[np.isfinite(x)]
                span = float(f.max() - f.min())
                if span <= 0:
                    continue
                for label, kw in every.items():
                    p = fluxprune.prune(t, x, **tuning_kwargs(kw, sig))
                    d = out[label].setdefault(sig, {"k": [], "capped": [], "rmse": [], "max": []})
                    d["k"].append(len(p))
                    d["capped"].append(not (p.rmse_met and p.max_error_met))
                    d["rmse"].append(p.rmse / span)
                    d["max"].append(p.max_error)
                    if label == "default":
                        sigma.setdefault(sig, []).append(p.noise)
    white = {m: [] for m in WHITE_SIGMAS}
    for day in range(DAYS):
        for t, x in blocks("white-noise", day, 1):
            for m in white:
                with warnings.catch_warnings():  # max_error = σ: below the noise on purpose
                    warnings.simplefilter("ignore")
                    white[m].append(len(fluxprune.prune(t, x, error_per_point=COARSE, max_error=float(m))))
    pyro = pyrometer_study({label: tuning_kwargs(kw, "pyrometer") for label, kw in every.items()})
    return out, sigma, white, pyro


def busiest_hour():
    _, _, centers = pyrometer(0)
    return int(np.argmax(np.bincount((centers // 3600).astype(int), minlength=24)))


def plot_tuning(group):
    """One hour of each example signal (day 0, 09:00; the pyrometer's busiest hour, minutes 10-30 shown) with the
    points each setting of a figure keeps: a row per signal, a column per setting."""
    settings = TUNING[group]
    fig, axes = plt.subplots(len(EXAMPLES), len(settings), figsize=(12, 1.5 * len(EXAMPLES) + 0.6),
                             layout="constrained", sharey="row", sharex=True)
    busy = busiest_hour()
    for row, sig in zip(axes, EXAMPLES):
        t, x = blocks(sig, 0, 1)[busy if sig == "pyrometer" else 9]
        f = x[np.isfinite(x)]
        span = f.max() - f.min()
        minutes = (t - t[0]) / 6e10
        window = (minutes >= 10) & (minutes <= 30) if sig == "pyrometer" else np.ones(len(t), bool)
        for col, (ax, (label, kw)) in enumerate(zip(row, settings.items())):
            kw = tuning_kwargs(kw, sig)
            p = fluxprune.prune(t, x, **kw)
            pm = (p.times - t[0]) / 6e10
            shown = (pm >= minutes[window][0]) & (pm <= minutes[window][-1])
            ax.plot(minutes[window], x[window], color=SCANS, lw=0.6, label="1 s scans")
            ax.plot(pm[shown], p.values[shown], "-o", color=C[0], lw=1.0, ms=2, label="kept points")
            if sig == "pyrometer":
                ax.set_yscale("symlog", linthresh=100)
            head = f"{label}\n" if ax in axes[0] else ""
            cap = "" if p.rmse_met and p.max_error_met else ", capped"
            ax.set_title(f"{head}{len(p)} points{cap} · RMSE {100 * p.rmse / span:.2g}% · max {p.max_error:.3g}",
                         loc="left", fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                guards = f"\nmax_error {MAX_ERROR[sig][0]:g} / {MAX_ERROR[sig][1]:g}" if group == "max-error" else ""
                ax.set_ylabel(sig + (" (min 10-30)" if sig == "pyrometer" else "") + guards, fontsize=8)
            tidy(ax)
    for ax in axes[-1]:
        ax.set_xlabel("minutes into the hour", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncols=2, frameon=False, fontsize=9)
    name = "examples" if group == "theta" else "examples-max-error"
    return save_fig(fig, OUT, f"{PREFIX}{name}", "One hour of eight signals with the points each setting keeps")


def tuning_table(out, group, extra=None):
    """Per signal: mean points per hour (the most in one block) and the share of blocks where the cap bound; below,
    the mean RMSE over the block's range · the largest error of any block in the signal's units."""
    names = SIGNALS + OFFSET + NOISY
    rows = []
    for sig in names:
        cells = [sig, *(extra(sig) if extra else [])]
        for label in TUNING[group]:
            d = out[label][sig]
            cap = f" · capped {100 * np.mean(d['capped']):.0f}%" if any(d["capped"]) else ""
            cells.append(f"{np.mean(d['k']):.0f} (max {max(d['k'])}){cap}<br><span class='muted'>"
                         f"{pct(np.mean(d['rmse']))} · {max(d['max']):.3g}</span>")
        rows.append(cells)
    total = ["<b>all</b>", *([""] * len(extra(names[0])) if extra else [])]
    for label in TUNING[group]:
        ks = [k for sig in names for k in out[label][sig]["k"]]
        cap = [c for sig in names for c in out[label][sig]["capped"]]
        e = [v for sig in names for v in out[label][sig]["rmse"]]
        total.append(f"<b>{np.mean(ks):.0f} (median {np.median(ks):.0f}) · capped {100 * np.mean(cap):.0f}%</b><br>"
                     f"<span class='muted'>{pct(np.mean(e))} (median {pct(np.median(e))})</span>")
    rows.append(total)
    head = ["signal", *(["σ", "loose", "fine"] if extra else []), *TUNING[group]]
    return table(head, rows)


def theta_section(tuning):
    out, _, _, pyro = tuning
    n = len(SIGNALS + OFFSET + NOISY)
    d = {label: [k for sig in out[label] for k in out[label][sig]["k"]] for label in TUNING["theta"]}
    over = 100 * np.mean(np.array(d["default"]) > CAP)
    body = (
        "<p><code>error_per_point</code> (θ) acts as an exchange rate between error and point count. Prune adds points "
        "in order of how much they reduce the time-weighted RMSE and stops once an additional point no longer saves θ of "
        "that RMSE (measured relative to the block's range and above its white-noise floor; see "
        "<a href='#noise'>the noise floor</a>). Pick θ first by taking the RMSE you would accept at some reference "
        "point count and dividing by that count, so θ = Y / X reads as \"Y error at X points.\" Note that X is not a "
        "fixed target: simpler blocks stop earlier, while busier blocks keep more points at a higher error. The default "
        "is 0.1% of the range at a tenth of the input length, θ = 0.001 / (n / 10), which is 2.8·10⁻⁶ for an hour of "
        "1 s scans. Adding <code>max_points</code> puts a hard ceiling on top by dropping the least significant points "
        "from the RMSE pass. We compare three settings below:</p><ul>"
        "<li><b>default</b> with no cap (equivalent to θ = 2.8·10⁻⁶ on these 1 h blocks);</li>"
        f"<li><b>default, max_points {CAP}</b>, capping the output at a tenth of the 1 s scans;</li>"
        f"<li><b>coarse</b>, with θ = {COARSE:g} (half the range at {CAP} points, or 5% at 36 points).</li></ul>"
        "<p>The figure shows one hour of eight signals (day 0 at 09:00, and minutes 10–30 of the pyrometer's busiest "
        "hour). Each panel reports the number of points kept in the hour, whether the cap bound, the RMSE as a "
        "percentage of the hour's range, and the worst single-scan error in the signal's units.</p>"
        + plot_tuning("theta")
        + f"<p>The table covers all {n} signals (<a href='#signals'>described below</a>) across 1 h blocks over all "
        f"{DAYS} days, listing mean points per hour (with the worst block's max), the fraction of blocks where the cap "
        "bound, and underneath, the mean RMSE over the block's range and the largest error across any block in signal "
        f"units. The default keeps an average of {np.mean(d['default']):.0f} points per hour (median "
        f"{np.median(d['default']):.0f}), and {over:.0f}% of blocks want more than {CAP} points—mostly noisy or "
        f"fast-moving signals that the cap holds to {CAP} at the cost of higher RMSE. The coarse θ averages "
        f"{np.mean(d['coarse']):.0f} points per hour and lets RMSE climb to several percent on busy signals, which "
        "mainly shows up in the worst-case error (bounded by <a href='#max-error'>max_error</a>).</p>"
        + tuning_table(out, "theta")
        + "<p>Below is the pyrometer scored against its underlying noise-free curve: points per hour in busy hours "
        "(with spikes) and quiet hours, points landing on spikes per hour, mean and worst peak shortfall (°C), and RMSE "
        "against the true curve (°C). Because time-weighted RMSE dilutes brief spikes (a 4 s spike of 600 °C adds only "
        "about 30 °C to an hour's RMSE), the coarse θ clips their peaks, whereas the default preserves them.</p>"
        + pyrometer_table({label: pyro[label] for label in TUNING["theta"]}))
    return body


def max_error_section(tuning):
    out, sigma, white, pyro = tuning
    w = {m: np.mean(v) for m, v in white.items()}
    fine, capped = list(TUNING["max-error"])[2:]
    ratio = {sig: max(out[capped][sig]["max"]) / MAX_ERROR[sig][1] for sig in out[capped]
             if any(out[capped][sig]["capped"])}
    lo, hi = min(ratio, key=ratio.get), max(ratio, key=ratio.get)
    body = (
        "<p><code>max_error</code> puts an upper bound on how far any scan can deviate from the reconstructed line, in "
        "the signal's own units. It only adds points: after the RMSE pass finishes, any segment that strays beyond the "
        "bound is split (see <a href='#guards'>the max_error pass</a>). This catches features that RMSE averages away, "
        "such as narrow spikes, glitches, or quiet blocks whose range is just dither. Set it relative to the sensor's "
        "noise σ (estimated by <code>Pruned.noise</code>), because lines through noisy scans cannot stay within a few σ "
        f"without retaining most of the data. On white noise at θ = {COARSE:g}, max_error = "
        + ", ".join(f"{m}σ keeps {w[m]:.0f}" for m in WHITE_SIGMAS)
        + " points per hour out of 3,600, and prune emits a warning if max_error is set below σ. When combined with "
        "<code>max_points</code>, the budget goes to max_error first and the RMSE pass uses whatever remains; if "
        "max_error cannot be met within the cap, it relaxes to roughly the tightest bound the cap permits (setting "
        "<code>max_error_met</code> to false). Below we test four settings on top of the coarse θ using per-signal "
        "tolerances (shown in the σ, loose, and fine columns):</p><ul>"
        f"<li><b>coarse</b>: θ = {COARSE:g} alone;</li>"
        "<li><b>loose max_error</b>: roughly the threshold an operator would notice;</li>"
        "<li><b>fine max_error</b>: around twice the sensor's noise, or the meaningful resolution for noise-free "
        "signals;</li>"
        f"<li><b>fine max_error, max_points {CAP}</b>: the same fine bound with a {CAP}-point safety cap.</li></ul>"
        + plot_tuning("max-error")
        + "<p>Results across all signals and 1 h blocks for all days, using the same layout as above; σ is the median "
        "noise estimate (0 where the signal has no noise or no regular cadence). A loose max_error only adds points "
        "where the coarse approximation missed real features, such as pyrometer spikes, furnace dropouts, and grid "
        "frequency swings. A fine bound around 2σ gets expensive on noisy signals, spending hundreds of points per hour "
        f"mostly tracing noise (temperature averages {np.mean(out[fine]['temperature']['k']):.0f}, white noise "
        f"{np.mean(out[fine]['white-noise']['k']):.0f}, and the pyrometer's quiet hours jump from around ten points to "
        f"over a hundred). The {CAP}-point cap reins them in by relaxing the bound—in the worst block, from "
        f"{ratio[lo]:.2f}× the target ({lo}) up to {ratio[hi]:.2f}× ({hi}). At 3σ or above, those same signals need "
        "far fewer points.</p>"
        + tuning_table(out, "max-error", lambda sig: [f"{np.median(sigma[sig]):.2g}", f"{MAX_ERROR[sig][0]:g}",
                                                       f"{MAX_ERROR[sig][1]:g}"])
        + "<p>Pyrometer metrics for the same four settings:</p>"
        + pyrometer_table({label: pyro[label] for label in TUNING["max-error"]}))
    return body


def plot_pyrometer():
    """A quiet hour and 20 minutes of a busy one of the pyrometer (day 0), with the points kept by the rmse pass
    without the noise floor, with it, and with it at a lower θ (each on its whole hour)."""
    x, truth, centers = pyrometer(0)
    counts = np.bincount((centers // 3600).astype(int), minlength=24)
    quiet, busy = int(np.argmin(counts)), int(np.argmax(counts))
    t = np.arange(3600.0)
    settings = (("θ = 0.0015, no floor", dict(error_per_point=0.0015, noise_floor=False)),
                ("θ = 0.0015, noise floor", dict(error_per_point=0.0015)),
                ("the default (θ = 2.8e-6 at 1 h), noise floor", {}))
    fig, axes = plt.subplots(3, 2, figsize=(12, 6.5), layout="constrained", sharex="col")
    for col, hour in enumerate((quiet, busy)):
        xb = x[3600 * hour:3600 * (hour + 1)]
        window = (t >= 600) & (t < 1800) if col else np.ones(3600, bool)
        for ax, (label, kw) in zip(axes[:, col], settings):
            p = fluxprune.prune(t, xb, **kw)
            shown = (p.times >= t[window][0]) & (p.times <= t[window][-1])
            ax.plot(t[window] / 60, xb[window], color=SCANS, lw=0.6, label="1 s scans")
            ax.plot(p.times[shown] / 60, p.values[shown], "-o", color=C[0], lw=1.0, ms=2.5, label="kept points")
            if col:
                ax.set_yscale("symlog", linthresh=100)
            ax.set_title(f"{label}: {len(p)} points in the hour", loc="left", fontsize=9)
            ax.tick_params(labelsize=8)
            tidy(ax)
        axes[-1, col].set_xlabel(f"minutes into {hour:02d}:00 ({'quiet' if col == 0 else 'busy, minutes 10-30'})",
                                 fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncols=2, frameon=False, fontsize=9)
    return save_fig(fig, OUT, f"{PREFIX}pyrometer", "A quiet and a busy hour of the pyrometer with the points kept")


def noise_section(noise):
    per, sweep_k, pyro = noise
    rows = []
    for name in SIGNALS + OFFSET + NOISY:
        d = per[name]
        rows.append([name, pct(np.median(d["sigma"]))]
                    + [f"{np.mean(d[off]):.0f} → {np.mean(d[on]):.0f}<br><span class='muted'>"
                       f"{pct(np.mean(d['e_' + off]))} → {pct(np.mean(d['e_' + on]))}</span>"
                       for off, on in (("off", "on"), ("doff", "don"))])
    body = ("<p>Straight lines through noisy samples always leave roughly σ of residual error until nearly every point "
            "is kept. On a quiet, noisy block whose range is only a few σ, that noise looks like real signal relative "
            "to the range, so the RMSE pass keeps spending points just to trace it. With the noise floor enabled (the "
            "default), the stopping criterion uses √(e² − σ²) instead of e—an estimate of the error against the true "
            "signal beneath the noise. σ is estimated in a single O(n) pass by taking each scan's distance from the "
            "chord connecting its two neighbors, normalizing by √(1 + w² + (1 − w)²) (where w is its relative time "
            "position between them), and scaling the median by 1.4826 (<code>fluxprune/_noise.py</code>). No smoothing "
            "is applied and the point ranking stays unchanged; only the stopping point moves. The table shows, across "
            "1 h blocks of all days, the median σ relative to block range, plus the points per hour (without → with "
            "the floor) and mean RMSE over range against the scans underneath, at both θ = 0.0015 (the old default) "
            "and the current default (θ = 0.001 / (n / 10), or 2.8e-6 for 3,600 scans).</p>"
            + table(["signal", "σ / range", "θ = 0.0015", "default"], rows))
    rows = [[tag, f"{np.mean(d['n']):.0f}", f"{100 * np.mean(np.array(d['sigma']) > 0):.0f}%",
             f"{np.mean(d['k']):.0f} · {pct(np.mean(d['e']))}", f"{np.mean(d['k_off']):.0f} · {pct(np.mean(d['e_off']))}",
             pct(np.mean(d["e_in"]))] for tag, d in per["archives"].items()]
    body += ("<h3>Archives</h3><p>Any noise estimator assumes sample times are independent of sample values, which is "
             "false for historian archives where swinging door or a deadband retains points specifically because of "
             "their values. Even on a smooth tag, archived points pass white-noise tests just like raw scans and yield "
             "an apparent σ near CompDev. To prevent false triggers, the noise floor only runs when a block has a "
             "regular cadence—at least 90% of intervals equal to one scan, using fluxcode's "
             "<code>fluxcode._noise.cadence</code> check. The table evaluates the tags' swinging-door archives over 1 h "
             "blocks across all days, showing archived points/h, the share of blocks activating the floor, and default "
             "prune points/h and RMSE over range against the 1 s scans (with and without <code>noise_floor</code>) "
             "alongside the archive's own RMSE. Tank level and valve are omitted because they only archive 2–5 points "
             "per hour.</p>"
             + table(["tag", "archived/h", "blocks with a floor", "default", "noise_floor=False", "the archive"], rows))
    zero = per["white-noise"]["zero"]
    body += (f"<p>Measuring against the underlying noise-free signal instead: on pure white noise (σ = 1 across "
             f"{len(zero[True])} hours), the reconstructed lines average {np.mean(zero[True]):.2f}σ RMSE from zero "
             f"with the floor versus {np.mean(zero[False]):.2f}σ without it, and {np.max(zero[True]):.2f}σ versus "
             f"{np.max(zero[False]):.2f}σ in the worst hour (with very few points kept, lines anchored at the required "
             "minimum and maximum lean slightly further).</p>")
    rows = [[name] + [f"{sweep_k[name, False][j]:.0f} → {sweep_k[name, True][j]:.0f}"
                      for j in range(len(NOISE_THETAS))] for name in NOISE_SHOWN]
    mean = [[np.mean([sweep_k[n, f][j] for n in SIGNALS]) for j in range(len(NOISE_THETAS))] for f in (False, True)]
    rows.append([f"<b>mean of the {len(SIGNALS)}</b>"] + [f"<b>{mean[0][j]:.0f} → {mean[1][j]:.0f}</b>"
                                                         for j in range(len(NOISE_THETAS))])
    body += ("<h3>Lower θ</h3><p>Points per hour on day 0 (without → with the floor) as θ decreases. Without the "
             "floor, tightening θ simply buys noise, so counts grow with the noise level. With the floor enabled, "
             "noise-dominated signals level off once their real structure is captured, and only signals with fine "
             "detail keep adding points.</p>"
             + table(["signal", *[f"θ = {th:g}" for th in NOISE_THETAS]], rows))
    n_spikes = len(next(iter(pyro.values()))["miss"])
    body += ("<h3>A pyrometer</h3><p>This signal motivated the noise floor: a pyrometer on a furnace exit conveyor "
             "sits on a noisy baseline most of the time, punctuated by narrow spikes (a / (b + t²), roughly 10 s wide) "
             f"as hot items pass—{n_spikes:,} spikes across {DAYS} days, separated by quiet hours between runs "
             "(<code>pyrometer</code> in <code>tslab/prune/signals.py</code>). Scored against its noise-free curve, "
             "the table lists points per hour in busy and quiet hours, points landing on spikes (more than 3σ above "
             "baseline), mean and worst peak shortfall (°C), and RMSE against the true curve (°C). The floor cuts "
             "quiet hours from tracing noise down to a handful of points. It does not change spike capture on its own, "
             "since time-weighted RMSE weights narrow spikes lightly (a 4 s spike of 600 °C adds only ~30 °C to an "
             "hour's RMSE); preserving spikes comes from using a tighter θ (like the default) or setting "
             "<code>max_error</code> to a few σ.</p>"
             + pyrometer_table(pyro) + plot_pyrometer())
    return body


GUARD_THETA = 0.0015  # the guard and refit studies' rmse pass: coarse, so that the guards do the work


WEEK_POINTS = 131_072  # a week-summary store: at most this many points per week ...
WEEK_THETA = 0.001 / WEEK_POINTS  # ... and 0.1% of the week's range at that many
HOUR = np.timedelta64(1, "h")  # for datetime64 times
HOUR_NS = 3_600 * 10**9  # for the corpus's integer nanoseconds


def intervals_study():
    """Per signal, day blocks of all days at a week-summary store's rate per hour (θ = 0.001 / 131,072 for a week,
    times 7 for a day; max_points 131,072 / 7), with and without include_regular_intervals = 1 h: points kept, how
    many of them forced and how many the rmse pass bought, and the RMSE."""
    warnings.filterwarnings("ignore", message="prune: max_points")
    out = {}
    for name in SIGNALS + OFFSET + NOISY:
        d = out.setdefault(name, {"plain": [], "hourly": [], "forced": [], "bought": [], "rmse": []})
        for day in range(DAYS):
            for t, x in blocks(name, day, 24):
                kw = {"error_per_point": WEEK_THETA * 7, "max_points": WEEK_POINTS // 7}
                a = fluxprune.prune(t, x, **kw)
                b = fluxprune.prune(t, x, include_regular_intervals=HOUR_NS, **kw)
                r = rank(t, x, HOUR_NS)
                d["plain"].append(len(a))
                d["hourly"].append(len(b))
                d["forced"].append(r.intervals)
                d["bought"].append(len(b) - r.min_points - (len(a) - rank(t, x).min_points))
                d["rmse"].append(b.rmse / a.rmse if a.rmse > 0 else 1.0)
    return out


def guard_study():
    """Per block of the main and offset signals (1 h): points, largest error and RMSE of the rmse pass, of
    max_error = the signal's GUARD, of the same with a cap of CAP points, and of swinging door with CompDev = GUARD."""
    items = scale_corpus(1)
    methods = ["rmse pass", "max_error", f"max_error, max_points {CAP}", "swinging door, CompDev = max_error / 2",
               "swinging door, CompDev = max_error"]
    out = {m: {"k": [], "max": [], "rmse": [], "capped": []} for m in methods}
    for it in items:
        r, x, t = it["r"], it["x"], it["t"]
        tol = GUARD[it["name"]]
        picks = {"rmse pass": r.select(GUARD_THETA), "max_error": r.select(GUARD_THETA, max_error=tol),
                 f"max_error, max_points {CAP}": r.select(GUARD_THETA, max_error=tol, max_points=CAP)}
        for m in methods:
            if m.startswith("swinging door"):
                idx = swinging_door((t - t[0]).astype(np.float64), x, tol / 2 if m.endswith("/ 2") else tol, np.inf)
                rmse_met = True
            else:
                idx, rmse_met = picks[m]
            worst = r.errors(idx)[1] * it["range"]
            # the cap cut the rmse pass, or relaxed max_error
            capped = "max_points" in m and (not rmse_met or worst > tol * (1 + 1e-9))
            out[m]["k"].append(len(idx))
            out[m]["max"].append(worst / tol)
            out[m]["rmse"].append(abs_error(it, idx) / SPAN[it["name"]])
            out[m]["capped"].append(capped)
    names = np.array([it["name"] for it in items])
    return names, {m: {key: np.array(v) for key, v in d.items()} for m, d in out.items()}


SPLITS = ["swinging door's rule", "farthest reach (fluxprune)", "fewest points (exact)"]


def max_error_pass_study():
    """Per block of the main and offset signals: the rmse pass's points (refined) with every line that strays further
    than max_error = GUARD split by each rule: points, RMSE over the span and the max_error pass's time."""
    items = scale_corpus(1)
    out = {m: {"k": [], "rmse": [], "ms": []} for m in SPLITS}
    for it in items:
        r = it["r"]
        if not len(r._tau):
            continue
        tol = GUARD[it["name"]] / it["range"]
        base = r.refine(r.top(r.count(GUARD_THETA)), PASSES)
        local = np.searchsorted(r._live, base)
        calls = {SPLITS[0]: lambda: B.other_max_error_pass(r._tau, r._y, local, tol, False),
                 SPLITS[1]: lambda: max_error_kernel(r._tau, r._y, local, tol),
                 SPLITS[2]: lambda: B.other_max_error_pass(r._tau, r._y, local, tol, True)}
        for m, call in calls.items():
            call()  # compiled and warm
            t0 = time.perf_counter()
            idx = r._live[call()]
            out[m]["ms"].append(1e3 * (time.perf_counter() - t0))
            out[m]["k"].append(len(idx))
            out[m]["rmse"].append(abs_error(it, idx) / SPAN[it["name"]])
    names = np.array([it["name"] for it in items if len(it["r"]._tau)])
    return names, {m: {key: np.array(v) for key, v in d.items()} for m, d in out.items()}


def max_error_pass_section(passes):
    f_names, f = passes
    rows = []
    for name in SIGNALS + OFFSET:
        sel = f_names == name
        rows.append([name] + [f"{d['k'][sel].mean():.0f} · {pct(d['rmse'][sel].mean())}" for d in f.values()])
    rows.append(["<b>all</b>"] + [f"<b>{d['k'].mean():.0f} · {pct(d['rmse'].mean())}</b><br><span class='muted'>"
                                  f"{1e3 * d['ms'].mean():.0f} µs per block, at most {1e3 * d['ms'].max():,.0f} µs</span>"
                                  for d in f.values()])
    return ("<h3>The max_error pass</h3><p>After the RMSE pass, any segment that strays beyond <code>max_error</code> "
            "is split starting from its left endpoint. Each subsequent scan constrains the upper and lower slope of a "
            "line from the current anchor that passes within <code>max_error</code> of that scan. <i>Swinging door's "
            "rule</i> terminates the segment at the first scan its chord cannot reach. <i>Farthest reach</i> (used by "
            "fluxprune) keeps scanning as long as the upper and lower slope bounds still overlap, tracking the farthest "
            "scan whose chord satisfies them; once the bounds cross, it makes that scan the next anchor. On noisy data, "
            "a later scan is often reachable even after an earlier one was not. <i>Fewest points</i> finds the exact "
            "minimum via dynamic programming over all feasible chords. Cells show points per hour and RMSE over the "
            "signal's span (starting from the same RMSE-pass points in every column), with per-block runtime for the "
            "max_error pass in the summary row.</p>"
            + table(["signal", *f.keys()], rows))


def refit_study():
    """Per block of the main and offset signals: the rmse pass's points with least-squares values (the required points
    pinned) against the samples: RMSE and largest error ratios; and with max_error = GUARD, how often it still holds."""
    out = {}
    for it in scale_corpus(1):
        r = it["r"]
        if not len(r._tau) or not np.isfinite(it["x"]).all():
            continue
        idx = r.select(GUARD_THETA)[0]
        e0, m0 = r.errors(idx)
        if e0 < 1e-9:  # already exact (the valve, the totalizer)
            continue
        e1, m1 = r.errors(idx, r.refit(idx))
        guarded = r.select(GUARD_THETA, max_error=GUARD[it["name"]])[0]
        held = r.errors(guarded, r.refit(guarded))[1] * it["range"] <= GUARD[it["name"]] * (1 + 1e-9)
        d = out.setdefault(it["name"], {"rmse": [], "max": [], "held": []})
        d["rmse"].append(e1 / e0)
        d["max"].append(m1 / m0)
        d["held"].append(held)
    return out


def timings():
    """Seconds, best of 3, by n (1 s scans, days concatenated): rank; prune with the rmse pass alone (rank, count,
    refine); prune with max_error = 1% of the range as well; and prune with refit=True. A temperature trace and white
    noise (whose max_error is below its noise on purpose: no warnings)."""
    warnings.filterwarnings("ignore", message="prune: max_error")
    rows = []
    for name in ("temperature", "white-noise"):
        x_all = np.concatenate([scans(name, d) for d in range(12)])
        for n in (3_600, 43_200, 86_400, 1_000_000):
            x = x_all[:n]
            t = (np.arange(n, dtype=np.int64) * 1_000_000_000).view("datetime64[ns]")
            tol = 0.01 * np.ptp(x)
            calls = {"rank": lambda: rank(t, x),
                     "prune": lambda: fluxprune.prune(t, x),
                     "max_error": lambda: fluxprune.prune(t, x, max_error=tol),
                     "refit": lambda: fluxprune.prune(t, x, refit=True),
                     "hourly": lambda: fluxprune.prune(t, x, include_regular_intervals=HOUR)}
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


def plot_curves(items):
    """Relative RMSE against points kept for the 09:00 block of day 0 of each signal, log-log."""
    picks = [it for it in items if it["day"] == 0 and it["block"] == 9]
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), layout="constrained", sharex=True)
    for ax, it in zip(axes.ravel(), picks):
        e = it["r"].relative_rmse()
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
    shown = ["count", "balance", "balance+noise", "marginal (RMSE)", "amortized", "amortized, no floor", "tolerance+noise"]
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


def table(head, rows, title=None):
    """Cells are HTML strings, or (class, HTML) pairs. Wrapped in a closed-by-default collapsible table."""
    def td(c):
        return f"<td class='{c[0]}'>{c[1]}</td>" if isinstance(c, tuple) else f"<td>{c}</td>"
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(td(c) for c in row) + "</tr>" for row in rows)
    return collapsible_table(title or f"Show data table ({len(rows)} rows)", f"<table><tr>{th}</tr>{body}</table>")


def shade(ratio):
    """Cell class for an error ratio against fluxprune."""
    return "" if ratio < 1.05 else "r1" if ratio < 1.25 else "r2" if ratio < 1.6 else "r3" if ratio < 2.5 else "r4"


# the page's order: (part anchor, part title, its lead, [section anchors]); a part without a title is its sections
PARTS = [
    ("", "", "", ["tuning", "max-error", "signals", "noise"]),
    ("how", "How it works, and how it was chosen",
     "<p>These sections cover the studies behind the defaults: ranking quality, stopping rules, error scaling, "
     "<code>max_error</code> enforcement, refitting, archive inputs, and runtime. Several experiments use prototype "
     "rules from <code>tslab/prune/rules.py</code>.</p>",
     ["quality", "curves", "rules", "scale", "guards", "refit", "intervals", "archives", "speed"]),
    ("", "", "", ["findings"]),
]


def write(items, params, res, sw, qual, opt, rate, length, scale, noise, tuning, guards, passes, refits,
          archive_res, archive_items, times, intervals):
    names = SIGNALS
    by = np.array([it["name"] for it in items])
    secs = {}

    def section(anchor, title, body):
        secs[anchor] = (title, body)

    n_blocks = len(items)
    intro = ("<p><b>fluxprune</b> selects a subset of points from a time series to be read back as piecewise-linear "
             "segments. It always keeps the first and last points (so adjacent blocks stitch cleanly) along with the "
             "minimum and maximum, then adds whichever points reduce the time-weighted RMSE most until another point "
             "no longer buys enough improvement. Calling <code>prune(times, values)</code> works out of the box with "
             "its defaults, and three parameters control what gets kept:</p><ul>"
             "<li><code>error_per_point</code> (θ) sets the exchange rate between error and point count—choose it "
             "first by dividing the relative RMSE you would accept at some reference point count by that count. The "
             "default is 0.1% of the block range at a tenth of the input points.</li>"
             "<li><code>max_points</code> enforces a hard upper limit on the number of kept points.</li>"
             "<li><code>max_error</code> bounds the worst single-scan error in the signal's own units, adding points "
             "wherever needed.</li></ul>"
             "<p>The first two sections show how these settings behave across signals; the rest of the report walks "
             "through the algorithm and the experiments behind each design choice.</p>"
             f"<p class='muted'>Dataset: {len(SIGNALS + OFFSET + NOISY)} synthetic signals × {DAYS} days of 1 s scans "
             "split into 1 h blocks of 3,600 points (<code>tslab/prune/signals.py</code>; the ranking and stopping-rule "
             f"studies use the first {len(names)} signals). Relative errors are normalized by each block's range "
             "(max − min). Regenerate with <code>uv run python make_prune_report.py</code> in "
             "<code>experimental/</code>.</p>")

    section("tuning", "Choosing θ and max_points", theta_section(tuning))
    section("max-error", "Adding max_error", max_error_section(tuning))
    rows = [[n, ("l", html.escape(describe(n)))] for n in SIGNALS + OFFSET + NOISY]
    section("signals", "The signals", "<p>The first eighteen signals form the main corpus for the ranking and "
            "stopping-rule studies. The next six test <a href='#scale'>error scaling</a> on small variations riding on "
            "large offsets, and the pyrometer tests <a href='#noise'>the noise floor</a>.</p>"
            + table(["signal", "what it is"], rows))

    # quality
    body = ("<p>Each method selects the same number of points per 1 h block (every 4th hour of day 0, giving 6 blocks "
            "per signal), and cells report the geometric mean of each method's RMSE relative to refined fluxprune. "
            "Both fluxprune and top-down include the required endpoints and extrema within k. For swinging door, "
            "CompDev is binary-searched to land at or just above k points. LTTB and min/max buckets are standard "
            "plotting downsamplers, while Visvalingam-Whyatt (cartographic line simplification) removes points "
            "bottom-up by triangle area; <i>+ refine</i> applies fluxprune's two refinement passes to its output.</p>")
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
    opt_rows = [[m] + [f"{gmean(opt[k][m]):.2f} (worst {max(opt[k][m]):.2f})" for k in QUALITY_KS]
                for m in opt[QUALITY_KS[0]]]
    body += ("<h3>Against the optimum</h3><p>On the same 3,600-point blocks, we compute the globally optimal k points "
             "(including the required ones) using O(n²k) dynamic programming—about 2 s per block, making it an oracle "
             "rather than a practical method. The table reports each greedy method's geometric-mean RMSE relative to "
             "that optimum.</p>"
             + table(["method", *[f"k = {k}" for k in QUALITY_KS]], opt_rows))
    section("quality", "Ranking quality at a fixed count", body)

    # curves
    section("curves", "Error curves", "<p>These curves show what a stopping rule sees: relative RMSE versus points "
            "kept (from the initial ranking, before refinement) for one hour of each signal. Smooth signals decay as "
            "k⁻² (sine), random walks decay as k⁻½ (temperature, pH), and white noise barely drops until most points "
            "are kept; step and ramp signals drop sharply and then flatten onto their noise floor. The dashed line "
            f"marks {pct(REF_EPS)} for reference.</p>" + plot_curves(items))

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
    body = (f"<p>Each rule's single parameter is calibrated so that all {n_blocks:,} blocks average {BUDGET} points per "
            "hour (1% of the 1 s scans), except knee detection, which cannot be calibrated and keeps whatever its α "
            "produces. Each signal cell shows mean points per hour above the geometric-mean block RMSE. The summary "
            "columns on the right report overall mean points per block, mean and median block RMSE, and the largest "
            "single block.</p>"
            + table(["rule", "what it does", "parameter", *names, "points/h", "mean", "median", "max k"], rows)
            + "<h3>Sublinear?</h3><p>Here each rule's per-block point count is plotted against the tolerance rule at "
            "the same total budget (the tolerance rule tracks complexity linearly by definition). A log-log fit gives "
            "the scaling slope: 0 corresponds to a fixed count, 1 to fixed tolerance, and 0.5 to their plain "
            "geometric mean. Fitted slopes: "
            + ", ".join(f"{rule} {s:.2f}" for rule, s in slopes.items()) + ".</p>" + img)
    sw_rows = [[rule] + [f"{pct(sw[b, rule]['mean'])} / {pct(sw[b, rule]['median'])}<br><span class='muted'>"
                         f"max k {sw[b, rule]['maxk']:,}</span>" for b in SWEEP_BUDGETS] for rule in RULES if rule != "knee"]
    body += ("<h3>Other budgets</h3><p>Repeating the same calibration across four target averages, showing mean / "
             "median block RMSE above the maximum points kept in any single block.</p>"
             + table(["rule", *[f"{b}/h" for b in SWEEP_BUDGETS]], sw_rows) + plot_sweep(sw))
    rate_rows = [[rule] + [f"{rate[rule][n]:.2f}" for n in names] for rule in RULES]
    length_rows = [[rule] + [f"{length[rule][n]:.2f}" for n in names] for rule in RULES]
    body += ("<h3>Sampling rate and block length</h3><p>A rule that adapts to the underlying signal rather than its "
             "storage layout should keep roughly as many points when an hour is sampled every 10 s as when sampled "
             "every second (ratio near 1 whenever both rates capture the signal, though white noise cannot be "
             "captured), and roughly as many in one 12 h block as across twelve 1 h blocks (count and balance scale "
             "K proportionally with block duration). Below is the median ratio of k at 10 s to k at 1 s across day 0's "
             "hours:</p>"
             + table(["rule", *names], rate_rows)
             + "<p>Ratio of k in a 12 h block to the sum of k across its twelve 1 h blocks (mean of the two 12 h "
             "halves on day 0):</p>"
             + table(["rule", *names], length_rows))
    section("rules", "How many points: the stopping rules", body)

    # scale
    s_items, s_names, study, slength = scale
    all_names = SIGNALS + OFFSET
    body = ("<p>So far, errors have been normalized by each block's range, which keeps relative precision independent "
            "of units and DC offset: a furnace drifting 4 °F around 2,115 °F receives the same relative precision as a "
            "valve swinging from 0 to 100%. Alternatives include normalizing by the block's largest absolute value, "
            "the geometric mean of range and max|x|, or a known per-series span (modeled here as each signal's range "
            "across all days, similar to a historian's configured span—something a schema-free store does not have). "
            "The choice of scale does not affect the point ranking, only the stopping thresholds. To test this, we add "
            "six offset <a href='#signals'>signals</a>: " + ", ".join(OFFSET) + ".</p>")
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
    body += (f"<p>Each rule-and-scale pair is calibrated to {BUDGET} points per hour across all {len(s_items):,} blocks "
             f"of the {len(all_names)} signals (5 days each). Cells report mean points per hour above the mean RMSE "
             "<i>relative to each signal's full span</i>, giving a unit- and block-independent error metric so the "
             "scales can be compared fairly (for the furnace with dropouts, error is measured on non-dropout samples). "
             "The final row shows the lowest possible mean RMSE over span at this budget, which requires knowing the "
             "span.</p>"
             + table(["rule", "errors relative to", *shown, "mean", "worst signal", "max k"], rows)
             + plot_scales(s_items, study))
    rows = [[rule, sc] + [f"{slength[rule, sc][n]:.2f}" for n in shown]
            + [f"<b>{gmean([slength[rule, sc][n] for n in shown]):.2f}</b>"] for rule, sc in slength]
    body += ("<p>Block-length scaling across all days: ratio of k in a 12 h block to the sum of k across its twelve "
             "1 h blocks, using θ / hours for the amortized rule. A ratio below 1 is partly expected because twelve "
             "separate blocks pay for twelve sets of required endpoints and extrema, and a longer block can allocate "
             "its points where they help most.</p>"
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
    section("noise", "The noise floor", noise_section(noise))

    section("guards", "max_error and max_points against swinging door",
            "<p>Here we compare the RMSE pass alone at θ = 0.0015 (the old coarse default, chosen so the guards do "
            "most of the work), with a per-signal <code>max_error</code> (each historian tag's CompDev, or an "
            f"equivalent practical tolerance), and with an added <code>max_points</code> = {CAP}/h cap (10% of the "
            "1 s scans). Any segment from the RMSE pass that exceeds <code>max_error</code> is split using the "
            "farthest-reach pass described below. For comparison, swinging door at CompDev = max_error / 2 provides "
            "the same worst-case guarantee, whereas at CompDev = max_error (last column) its lines can stray up to "
            "twice the tolerance because the doors pivot around the archived point. Each cell shows mean points/h · "
            "worst block error divided by max_error · mean RMSE over the signal's span, with the percentage of capped "
            "blocks (where <code>rmse_met</code> or <code>max_error_met</code> is false) underneath.</p>"
            + table(["signal", "max_error", *g.keys()], rows)
            + max_error_pass_section(passes))

    # refit
    rows = [[name, f"{gmean(d['rmse']):.2f}", f"{gmean(d['max']):.2f}", pct(np.mean(d["held"]), 3)]
            for name, d in refits.items()]
    every = [v for d in refits.values() for v in d["rmse"]]
    rows.append(["<b>all</b>", f"<b>{gmean(every):.2f}</b>",
                 f"<b>{gmean([v for d in refits.values() for v in d['max']]):.2f}</b>",
                 f"<b>{pct(np.mean([v for d in refits.values() for v in d['held']]), 3)}</b>"])
    section("refit", "Refit: values that aren't samples",
            "<p><code>prune(refit=True)</code> keeps the same timestamps but solves an O(n) tridiagonal system to "
            "replace the interior values with the best time-weighted L2 piecewise-linear fit, while leaving the "
            "required endpoints, extrema, and gap-boundary points pinned to their original samples. Across all 1 h "
            "blocks with non-zero error, the table reports the geometric-mean RMSE and max-error ratios (refit vs "
            "raw samples) at the RMSE pass's points, along with how often the signal's <code>max_error</code> guard "
            "still holds after refitting (since refit does not re-check the bound).</p>"
            + table(["signal", "RMSE, refit / samples", "largest error, refit / samples", "max_error still holds"], rows))

    # regular intervals
    rows = []
    hours = DAYS * 24

    def per_hour(v):
        return f"{np.sum(v) / hours:.0f}"

    for name, d in intervals.items():
        rows.append([name, per_hour(d["plain"]), per_hour(d["hourly"]), per_hour(d["forced"]),
                     f"{np.sum(d['bought']) / hours:+.1f}", f"{gmean(d['rmse']):.3f}"])
    every = {key: [v for d in intervals.values() for v in d[key]] for key in ("plain", "hourly", "forced", "bought")}
    n_sig = len(intervals)
    rows.append(["<b>mean</b>", *(f"<b>{np.sum(every[key]) / hours / n_sig:.0f}</b>"
                                  for key in ("plain", "hourly", "forced")),
                 f"<b>{np.sum(every['bought']) / hours / n_sig:+.1f}</b>",
                 f"<b>{gmean([v for d in intervals.values() for v in d['rmse']]):.3f}</b>"])
    section("intervals", "Regular intervals: each hour's first, minimum and maximum",
            "<p>Passing <code>include_regular_intervals=np.timedelta64(1, \"h\")</code> forces prune to retain the "
            "first point, minimum, and maximum of every clock hour so downstream queries can read exact hourly stats "
            f"directly from the pruned series. Here we test day-long blocks across all {DAYS} days using a "
            f"week-summary store's rate: up to {WEEK_POINTS:,} points per week and θ = 0.001 / {WEEK_POINTS:,} for a "
            f"week (scaled to θ × 7 and at most {WEEK_POINTS // 7:,} points per day). The table lists points/h without "
            "and with hourly intervals, forced points/h beyond the block's own required points, the net change in "
            "points bought by the RMSE pass, and the geometric-mean RMSE ratio. Forced points do not count against "
            "the exchange rate, so on most signals the RMSE pass buys about the same or slightly fewer points. The "
            "exception is a quiet block sitting on its noise floor: normally its lines already sit near σ and the "
            "pass stops immediately, but pinning hourly minima and maxima to noise extremes nudges the error slightly "
            "above σ, prompting the pass to add points across the block until it drops back to the floor (bounded by "
            "the cap).</p>"
            + table(["signal", "points/h", "points/h, hourly", "forced/h", "bought/h", "RMSE, hourly / without"],
                    rows))

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
            "<p>Applying the same calibrated rules to each tag's swinging-door historian archive "
            "(<code>tslab/common/historian.py</code>, using exception plus swinging-door filtering at the tag's "
            "CompDev) in 1 h blocks. Cells show points kept per hour (and the percentage of archived points retained), "
            "above the geometric-mean RMSE against the archive's own piecewise-linear trend relative to block "
            "range.</p>"
            + table(["tag", "archived/h", "count", "tolerance", "balance", "tolerance+noise", "balance+noise"], rows))

    # speed
    rows = []
    for name, n, b in times:
        tr = b["rank"]
        rows.append([name, f"{n:,}", f"{1e3 * tr:.1f} ms", f"{1e9 * tr / n:.0f} ns"]
                    + [f"+{1e3 * (b[key] - tr):.1f} ms (+{100 * (b[key] - tr) / tr:.0f}%)"
                       for key in ("prune", "max_error", "refit", "hourly")])
    power = ("" if "AC Power" in subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True).stdout
             else "<b>Measured on battery power: indicative only, rerun on AC.</b> ") if sys.platform == "darwin" else ""
    section("speed", "Speed", f"<p>{power}Single-threaded timings on an Apple M3 (best of 3 runs on 1 s scans). "
            "<code>rank</code> is the O(n log n) heap pass; <code>prune</code> adds the O(n) stopping selection and "
            "two O(n) refinement passes, reported as time added over <code>rank</code>. <code>max_error</code> (at 1% "
            "of range), <code>refit</code> (least-squares values), and <code>hourly</code> "
            "(<code>include_regular_intervals</code> = 1 h) are likewise shown as overhead above <code>rank</code>. "
            "See <code>bench/prune.py</code> for per-call timings across more signals.</p>"
            + table(["signal", "n", "rank", "per point", "prune (rmse pass)", "prune, max_error", "prune, refit",
                     "prune, hourly"], rows))

    section("findings", "Findings", FINDINGS)
    toc, sections = [], []
    for part, title, lead, anchors in PARTS:
        level = "h3" if title else "h2"
        subs = [(a, secs[a][0]) for a in anchors]
        if title:
            toc.append((part, title, subs))
            sections.append(f"<h2 id='{part}'>{html.escape(title)}</h2>\n{lead}")
        else:
            toc += [(a, t, []) for a, t in subs]
        for a in anchors:
            t, body = secs.pop(a)
            if title:
                body = body.replace("<h3", "<h4").replace("</h3>", "</h4>")
            sections.append(f"<{level} id='{a}'>{html.escape(t)}</{level}>\n{body}")
    assert not secs, f"sections not on the page: {list(secs)}"
    (OUT / "prune.html").write_text(shell("fluxprune: keeping the points that matter", intro, toc, "\n".join(sections), active="prune"))


FINDINGS = """
<ol>
<li><b>Bottom-up ranking plus refinement gets close to optimal.</b> Repeatedly removing the cheapest point under exact
time-weighted L2 using a heap (O(n log n)) produces a nested ranking and the full error curve in one pass, though greedy
merging alone sits about 30% above the global optimum. Two O(n) refinement passes—sliding each kept point to the best
position between its neighbors—close that gap to 4–6% above optimal on 1 h blocks at 12, 36, and 108 points (worst case
1.4×) while adding only 10–30% to ranking time. Top-down splitting (the Douglas-Peucker / PIP order) beats plain
bottom-up on step-like signals but can blow up badly elsewhere (up to 74× optimal on the compressor).
Visvalingam-Whyatt (bottom-up by triangle area) yields 1.6–1.8× fluxprune's RMSE on its own and 1.03–1.07× after the
same refinement passes, confirming that refinement does most of the heavy lifting while exact L2 still provides a better
starting order. SWAB (Keogh et al. 2001) runs bottom-up inside a sliding window for streaming, which reduces to plain
bottom-up on full blocks. At matched point counts, swinging door has 1.3–1.5× fluxprune's RMSE, LTTB has 2–2.5×, and
min/max buckets and uniform spacing have 2–4×, since plotting downsamplers preserve visual silhouette rather than linear
interpolation accuracy.</li>
<li><b>The error curve separates signal from noise.</b> Smooth signals decay as k⁻², random walks decay as k⁻½, and
white noise stays flat until nearly all points are kept. Evaluating the curve at k = n/2 gives a clean white-noise
estimate of 0.29σ across all block sizes (<code>noise</code> in <code>tslab/prune/rules.py</code>), whether timestamps
are regular or irregular.</li>
<li><b>Neither fixed count nor fixed tolerance works for mixed data.</b> Fixed count spends the same budget on a valve
that needs two points as on pure white noise. Fixed tolerance fails in the opposite direction: calibrated to a 36/h
average, it pours budget into noisy blocks (194 points/h on white noise, up to 432 in one block) and pushes ε up to 13%
of range for everything else, yielding the worst mean error of any rule.</li>
<li><b>Geometric-mean balancing helps, especially with a noise floor.</b> Taking the geometric mean of fixed count and
fixed tolerance (<code>balance</code>, √(K·k_ε)) produces a genuine compromise (scaling slope ~0.6 against tolerance,
bounded by √(K·n)), but white noise still reads as the most "complex" signal and takes 4× the average budget. Raising ε
to each block's noise level (<code>balance+noise</code>) fixes that, nearly matching the lowest possible mean RMSE at
36/h (3.4% vs 3.3%) with the smallest worst-case block among adaptive rules (132 points). At high budgets it
intentionally levels off rather than tracing noise, though correlated noise (flow with τ = 5 s at 1 s scans) partially
reads as signal and makes point counts sensitive to sampling rate (0.25× at 10 s).</li>
<li><b>Marginal stopping rules break under resampling.</b> Stopping when a point saves less than λ achieves the lowest
possible mean error for a fixed total budget by construction (minimizing e + λk), but a noise point's marginal gain
scales as 1/n. As a result, the same λ keeps 13–25× more noise points at 10 s sampling than at 1 s, and on noisy data
marginal rules tend to either quit immediately or keep almost everything.</li>
<li><b>The amortized rule is the best single-knob default.</b> Stopping at the first k where remaining error per added
point drops below θ (exposed as <code>error_per_point</code>) stays within 4–21% of the lowest possible mean RMSE at
every budget without a noise floor, caps the largest block at 2.4–3.7× the average, and is the most stable across
sampling rates (0.80–1.00× across all signals). Geometrically, it stops where the error curve crosses the line θ·k,
trading error for points at a constant rate so counts grow sublinearly with complexity (k ∝ c^⅓ for smooth k⁻² curves,
and k ∝ σ for noise).</li>
<li><b>Adding the noise floor makes low θ practical.</b> By evaluating the amortized rule on √(e² − σ²)—estimating σ in
one O(n) pass from the median distance of each scan to its neighbors' chord—prune stops spending points on white noise.
At θ = 0.0015 it slashes points on noisy blocks (white noise 95 → 11/h, motor current 67 → 12/h, quiet level 77 → 19/h,
vibration 47 → 18/h) while leaving smooth and quantized signals unchanged, at a cost of 0–10% higher RMSE against the
noisy scans. Calibrated to 36/h it matches the optimal mean RMSE (3.3%, tying <code>marginal (RMSE)</code>) with half
the median error (0.95% vs 1.7%, compared to 3.6% mean without the floor), and at 324/h it levels off at 2.8% vs 1.9%
rather than chasing noise. Lowering θ no longer wastes points on noise: at θ = 1.5·10⁻⁶, white noise stays at 17/h
instead of 3,142/h while smooth signals keep refining. On the pyrometer, the floor trims quiet hours from 94 to 11
points at θ = 0.0015 (improving true-curve RMSE from 0.66 °C to 0.56 °C), while narrow spikes are preserved by a tighter
θ (default: 236 points/busy hour, 22/quiet hour, every peak within 7 °C) or <code>max_error</code> = 10σ (162 points,
same peaks). At the default θ, the floor is essential: without it, noisy signals keep most of their scans (white noise
2,941/h, motor current 2,705/h, pyrometer quiet hours 2,933/h) versus 18, 22, and 22 with it. There are two trade-offs:
correlated noise still looks partly like signal at fast scan rates (making flow drop to 0.14× at 10 s scans, and
temperature/pH to 0.66–0.69×), and on pure white noise the few remaining lines anchored at the required min and max lean
slightly further (worst of 120 hours is 0.48σ from zero vs 0.38σ without the floor, though the mean improves from 0.33σ
to 0.26σ). Because value-dependent archive timestamps violate the white-noise assumption—causing SDT archives to look
like white noise with σ ≈ CompDev (3× raw noise on temperature) and doubling scan RMSE from 1.9% to 3.3%—the floor only
activates when at least 90% of intervals match the scan cadence (using fluxcode's cadence test), leaving SDT archives
untouched. Scoring against a smoothed copy of the scans was also tested and rejected because chords between noisy
samples are no closer to a smooth curve, requiring a tuning window and shaving narrow peaks.</li>
<li><b>Knee detection is not viable.</b> Stopping when doubling the point count stops paying off fails on both ends:
periodic signals hit early plateaus (4 points for 12 h of a sine wave), while random walks never stop (3,061 of 3,600
points on temperature).</li>
<li><b>Longer blocks keep fewer points per hour.</b> Every rule except fixed count uses fewer points in one 12 h block
than in twelve 1 h blocks. With θ / hours for the amortized rule (analogous to K · hours for count), the geometric-mean
ratio across the 18 signals is 0.30 (ranging from 0.03 to 0.90, lowest on quiet-level/dropout signals where one event
sets the range and on white noise where the floor holds counts roughly constant). Wider block ranges account for part of
the drop—using a known per-series span raises the ratio to 0.55—and the rest is expected, since twelve blocks store
twelve sets of required endpoints/extrema while one long block allocates points where they matter most.</li>
<li><b>Block range is the best schema-free error scale.</b> Calibrated to 36/h with the amortized rule, normalizing by
block range yields 1.2% mean RMSE over signal span, close to the 1.0% achieved with known spans (and the 1.0% theoretical
minimum). It automatically gives small variations on large offsets appropriate precision: furnace (4 °F/h around
2,115 °F) and grid frequency keep 29 and 100 points/h, close to 23 and 126 with known spans. Normalizing by
<b>max|x|</b> depends arbitrarily on where zero sits (ambient temperature keeps 74 points/h in °C vs 14 in kelvin) and
starves high-offset signals (furnace and grid frequency drop to 9 and 8 points/h, pushing error to 1.2% and 4.7% of span
vs 0.32% and 1.5%). Using <b>√(range · max|x|)</b> cuts both distortions in half without eliminating them (64 vs 26
points for °C vs K, grid frequency 2.7% of span) and improves block-length scaling less than a known span (0.40 vs
0.55). Block range has two weaknesses—quiet blocks treat dither as signal (37 points/h tracing a 0.01 step on
<code>level-quiet</code> vs 6 with known span, or 77 without the noise floor) and glitches widen the range and coarsen
precision (<code>furnace-dropout</code> rises to 1.1% of span vs 0.32%)—both of which are solved by a known span or an
absolute <code>max_error</code> guard.</li>
<li><b>Swinging-door archives prune further depending on noise.</b> Tags whose CompDev sits below their noise floor
(flow and motor current, archiving 1,300–3,000 points/h) shrink to 0.6–6% of the archive at the same settings; smooth
tags (temperature and pH at ~330/h) shrink to 2–11%; and already-sparse archives (~2 points/h) pass through unchanged.</li>
<li><b><code>max_error</code> and <code>max_points</code> outperform swinging door.</b> The <code>max_error</code> pass
splits any segment exceeding the tolerance using farthest reach: extending from each anchor to the farthest scan whose
chord keeps every intermediate scan within <code>max_error</code>. Standard swinging door (stopping at the first
unreachable scan) keeps 1.1–5× as many points on noisy signals (vibration 176 vs 36/h, batch 73 vs 23/h, temperature 766
vs 509/h), while exact dynamic programming saves only 4% more points at roughly 38× the runtime (490 µs vs 13 µs per
block, up to 8.4 ms). At the same worst-case guarantee as swinging door (CompDev = max_error / 2, since CompDev =
max_error allows 1.6–2× excursions), fluxprune uses fewer points on 12 of 18 signals (batch 23 vs 228/h, vibration 36 vs
422/h, temperature 509 vs 1,136/h, grid frequency 282 vs 597/h), ties on two, and uses more only where the RMSE pass
already wanted extra points (19 vs 2 on <code>level-quiet</code>'s dither). Setting <code>max_error</code> below the
noise level forces any method to keep most scans (3,160/h on motor current; roughly half at σ, two-thirds at 0.6σ, and
one-eighth at 2.5σ), which is why <code>prune()</code> warns below σ. Adding <code>max_points</code> (360/h here)
allocates points in strict priority: required points first, then <code>max_error</code> if feasible within the cap (with
the RMSE pass taking any leftover budget via bisection on its least significant points), or else relaxing
<code>max_error</code> via bisection to the tightest bound the cap allows with zero RMSE-pass points. The
<code>max_error</code> pass always runs last because inserting a point can increase error elsewhere. The cap only bound
on the five noise-dominated signals, relaxing the worst block to 1.2× max_error on temperature, 1.3× on pH, 2.1× on
white noise, 2.7× on flow, and 11× on motor current (where σ is 5× max_error). Alternatives tested and rejected:
splitting the worst segment at its worst point under a cap (1.3–1.7× higher error from anchoring on noise spikes),
running <code>max_error</code> before ranking with its points pinned (same RMSE at equal count, 3–5× slower), secant
search for the relaxed bound (9.4 passes mean vs 12.8, but up to 28 vs 14 worst-case), and sweeping every RMSE-pass
count instead of bisecting (only 0.4% lower mean RMSE).</li>
<li><b>Refit (<code>refit=True</code>) helps smooth signals but breaks <code>max_error</code>.</b> Solving an O(n)
tridiagonal least-squares fit for interior values (keeping required points pinned) cuts RMSE by roughly 10% overall—most
of it on smooth or step-like signals (sine, compressor, batch at 0.5–0.8×) where chords clip corners, and only 1–3% on
noisy signals. Because refit values are no longer original readings and least-squares lines shift away from scans that
defined the tolerance, <code>max_error</code> continues to hold in only 41% of blocks (0% on noisy signals, 100% on
smooth or quiet ones). It remains an optional flag for evaluation on real data; joint time-and-value optimization (such
as free-knot splines or Imai–Iri 1986 fewest-segments) could improve further and was not tested.</li>
</ol>
<p><b>API summary.</b> <code>prune(t, x)</code> returns the kept timestamps and values, the resulting RMSE and max
error, and the status flags <code>rmse_met</code>, <code>max_error_met</code>, and <code>max_points_met</code>. By
default it runs the RMSE pass with <code>noise_floor=True</code> and <code>error_per_point</code> (θ) set to 0.1% of
block range at a tenth of the input length, θ = 0.001 / (n / 10) (2.8·10⁻⁶ for 1 h of 1 s scans, keeping ~4% of scans on
average and 2% on the median block). Scaling θ by input length keeps density consistent across longer blocks, though a
slower sampling rate keeps fewer points per hour. Two optional guards layer on top: <code>max_points</code> sets a hard
ceiling, and <code>max_error</code> caps single-scan error in signal units (handling quiet blocks, glitches, and long
blocks for configured tags). Setting θ = 0 with <code>max_points</code> keeps the best <code>max_points</code> points,
while θ = ∞ with <code>max_error</code> runs only the <code>max_error</code> pass. Runs of non-finite values always
retain their first point and the adjacent finite points (forward hold), even past <code>max_points</code> if necessary.
The other stopping rules evaluated on this page (count, tolerance, balance, noise-raised ε, marginal, and knee) are
experimental baselines in <code>tslab/prune/rules.py</code>.</p>
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
    noise = noise_study()
    tuning = tuning_study()
    print(f"noise and tuning: {time.time() - t0:.0f} s", flush=True)
    guards = guard_study()
    passes = max_error_pass_study()
    refits = refit_study()
    intervals = intervals_study()
    print(f"guards and refit: {time.time() - t0:.0f} s", flush=True)
    archive_items = corpus(archive=True, names=TAGS)
    archive_res = apply_rules(archive_items, params)
    times = timings()
    print(f"done: {time.time() - t0:.0f} s", flush=True)
    write(items, params, res, sw, qual, opt, rate, length, scale, noise, tuning, guards, passes, refits,
          archive_res, archive_items, times, intervals)


if __name__ == "__main__":
    main()
