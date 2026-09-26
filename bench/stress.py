# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Synthetic day-scale stress test of a sensor fleet: T tags x U units x 60 blocks x 1000 samples through the public API,
encode then decode, on W workers. The default is the full target: 1000 tags at 1 kHz for a
day, in 1-minute units of 1-second blocks (86.4e9 samples, 691 GB of float64).

Generating 691 GB of synthetic signal would take far longer than coding it, so each signal kind
has a pool of --pool distinct pre-generated units (default 32 x 480 KB per kind, well past the
caches) and tag t's unit u is pool[(t * U + u) % pool] of the tag's kind. The codec keeps no state
between units, so the work per unit is what distinct data would cost. NaN runs are laid out per
tag and day (--nan-minutes in --nan-runs runs at random, non-block-aligned sample offsets) and
written over a copy of the pool unit in the worker; that copy is outside the per-call timings but
inside the wall time. --sprinkle bakes isolated NaNs into every pool unit (the slow path on every
block: the adversarial case).

Timed: encode_unit per unit (index columns and zstd included), then decode_unit per unit. The
decode phase reads the encoded pool (plus pre-encoded NaN units), so it does the same per-unit
work as decoding the day. Workers claim whole tags (a day each) from a shared counter. A roundtrip check on every pool unit runs first.

    uv run python bench/stress.py                                  # full target, sensor-mix, 4 threads
    uv run python bench/stress.py --scale 0.05                     # 5% of the tags (quick)
    uv run python bench/stress.py --data random-walk --mode processes
    uv run python bench/stress.py --data "analog:3,sensor-0.1:1" --sprinkle 0.001
    uv run python bench/stress.py --list                           # signal kinds
"""

import argparse
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from _signals import DISCRETE, KINDS, MINUTE, discrete_minute, minute

import fluxcode
from fluxcode import Params, _api, _decoder, _format

BLOCK, BLOCKS = 1000, 60
DAY_UNITS = 1440


# --- Sensor-fleet signals (one minute each; the rest come from tests/_signals.py) ---

def _fleet(kind, rng):
    n = MINUTE
    if kind == "analog":  # process variable: first-order lag after setpoint moves, sensor noise, 16-bit ADC
        sp = np.repeat(rng.normal(50, 5, 8), n // 8 + 1)[:n]
        a = np.exp(-1 / 3000)
        pv = np.empty(n)
        v = sp[0]
        for i in range(0, n, 1000):  # the lag per 1000 samples, exact enough and fast
            seg = sp[i:i + 1000]
            k = np.arange(1, seg.shape[0] + 1)
            pv[i:i + seg.shape[0]] = seg + (v - seg) * a ** k
            v = pv[i + seg.shape[0] - 1]
        adc = 100 / 65536
        return np.round((pv + rng.normal(0, 0.05, n)) / adc) * adc
    if kind == "held":  # report-by-exception / deadband: held values, 0.01 decimal, new value every ~0.2-5 s
        cuts = np.sort(rng.integers(0, n, rng.integers(12, 300)))
        levels = np.round(np.cumsum(rng.normal(0, 0.3, cuts.shape[0] + 1)) + rng.uniform(0, 500), 2)
        return levels[np.searchsorted(cuts, np.arange(n), side="right")]
    if kind == "digital":  # 0/1 state, rare transitions
        cuts = np.sort(rng.integers(0, n, rng.integers(0, 6)))
        return (np.searchsorted(cuts, np.arange(n), side="right") % 2).astype(np.float64)
    if kind == "vibration":  # accelerometer: harmonics of 29.5 Hz plus broadband noise, stored as float32
        t = np.arange(n) / 1000 + rng.uniform(0, 1e4)
        x = sum(a * np.sin(2 * np.pi * 29.5 * h * t + rng.uniform(0, 6.3)) for h, a in ((1, 1.0), (2, 0.4), (3, 0.15)))
        return (x + rng.normal(0, 0.2, n)).astype(np.float32).astype(np.float64)
    if kind == "counter":  # totalizer: integral of a noisy flow, 0.001 decimal
        return np.round(rng.uniform(0, 1e5) + np.cumsum(np.abs(rng.normal(2e-3, 3e-4, n))), 3)
    raise ValueError(kind)


FLEET = ["analog", "held", "digital", "vibration", "counter"]
DISCRETE_NAMES = [d[0] for d in DISCRETE]
ALL_KINDS = FLEET + KINDS + DISCRETE_NAMES
# A guess at a sensor fleet's tag mix: mostly slow analog and held values, a few fast/noisy channels.
PRESETS = {
    "sensor-mix": {"analog": 35, "held": 25, "digital": 10, "vibration": 10, "counter": 5,
                "sensor-0.1": 5, "noisy-sine": 5, "random-walk": 5},
    "all": {k: 1 for k in ALL_KINDS},
}


def gen_minute(kind, seed):
    if kind in FLEET:
        return _fleet(kind, np.random.default_rng(seed))
    if kind in DISCRETE_NAMES:
        return discrete_minute(kind, seed)
    return minute(kind, seed)


def parse_data(spec):
    if spec in PRESETS:
        return PRESETS[spec]
    mix = {}
    for part in spec.split(","):
        name, _, w = part.partition(":")
        if name not in ALL_KINDS:
            raise SystemExit(f"unknown kind {name!r}; --list shows them")
        mix[name] = float(w or 1)
    return mix


# --- the workload: which pool unit and which NaN spans each (tag, unit) gets ---

class Workload:
    def __init__(self, a):
        self.a = a
        self.params = Params()
        self.mix = parse_data(a.data)
        self.tags, self.units = a.tags, a.units
        rng = np.random.default_rng(a.seed)
        kinds, w = list(self.mix), np.array(list(self.mix.values()), float)
        # deterministic proportional assignment (largest remainder), shuffled across tags
        counts = np.floor(w / w.sum() * self.tags).astype(int)
        for i in np.argsort(-(w / w.sum() * self.tags - counts))[:self.tags - counts.sum()]:
            counts[i] += 1
        self.tag_kind = rng.permutation(np.repeat(np.arange(len(kinds)), counts))
        self.kinds = kinds
        self.nan_spans = self._nan_spans(rng)
        self.pool: list[np.ndarray] = []  # filled by build_pool
        self.enc: list[list[bytes]] = []

    def _nan_spans(self, rng):
        """{(tag, unit): [(start, stop), ...]} sample spans within the unit, from runs laid out over
        each tag's day at arbitrary sample offsets (so rarely on block or unit boundaries)."""
        a, spans = self.a, {}
        day = self.units * MINUTE
        total = a.nan_minutes * MINUTE * self.units / DAY_UNITS
        if total <= 0 or a.nan_runs <= 0:
            return spans
        for t in range(self.tags):
            lens = rng.dirichlet(np.ones(a.nan_runs)) * total
            for L in lens.astype(np.int64):
                s = int(rng.integers(0, max(day - L, 1)))
                e = min(s + int(L), day)
                while s < e:
                    u, off = divmod(s, MINUTE)
                    stop = min(e - u * MINUTE, MINUTE)
                    spans.setdefault((t, u), []).append((off, stop))
                    s = u * MINUTE + stop
        return spans

    def build_pool(self):
        """Pool units per kind, with any --sprinkle applied, and their encodings for the decode phase."""
        a = self.a
        rng = np.random.default_rng(a.seed + 1)
        self.pool, self.enc = [], []
        for ki, k in enumerate(self.kinds):
            xs = np.stack([gen_minute(k, 1_000_000 * ki + j) for j in range(a.pool)])
            if a.sprinkle > 0:
                xs[rng.random(xs.shape) < a.sprinkle] = np.nan
            self.pool.append(xs)
            self.enc.append([fluxcode.encode_unit(x, self.params).unit for x in xs])
        self.nan_enc = {tu: fluxcode.encode_unit(self.unit(*tu)[0], self.params).unit for tu in self.nan_spans}

    def slot(self, t, u):
        return self.tag_kind[t], (t * self.units + u) % self.a.pool

    def encoded(self, t, u):
        e = self.nan_enc.get((t, u))
        if e is None:
            k, j = self.slot(t, u)
            e = self.enc[k][j]
        return e

    def unit(self, t, u):
        """(samples, is a private copy with NaN spans)"""
        k, j = self.slot(t, u)
        x = self.pool[k][j]
        sp = self.nan_spans.get((t, u))
        if sp is None:
            return x, False
        x = x.copy()
        for s, e in sp:
            x[s:e] = np.nan
        return x, True


def check_pool(wl):
    """Roundtrip every pool unit: NaN positions exact, finite error within the step's half (<= range/2^6/2)."""
    worst = 0.0
    for k, (xs, encs) in enumerate(zip(wl.pool, wl.enc)):
        for x, u in zip(xs, encs):
            y = fluxcode.decode_unit(u)
            nx = ~np.isfinite(x)
            assert np.array_equal(nx, ~np.isfinite(y)), f"{wl.kinds[k]}: non-finite positions differ"
            X, Y, F = x.reshape(BLOCKS, BLOCK), y.reshape(BLOCKS, BLOCK), ~nx.reshape(BLOCKS, BLOCK)
            for b in range(BLOCKS):
                f = F[b]
                if f.any():
                    r = X[b][f].max() - X[b][f].min()
                    err = np.abs(X[b][f] - Y[b][f]).max()
                    if err > 0:
                        assert err <= r / 2 ** 6, f"{wl.kinds[k]}: error {err} over range {r}"
                        worst = max(worst, err / r)
    return worst


# --- workers ---

def _claim(counter):
    with counter.get_lock():
        t = counter.value
        counter.value += 1
    return t


def _work(wl, go, switch, counters, out):
    """Claim tags one at a time (kinds differ in cost, so a static split leaves workers idle at the
    end): encode each claimed tag's day; after every worker is done, decode the same way."""
    p = wl.params
    enc_t = dec_t = 0.0
    nbytes = nan_units = 0
    go.wait()
    while (t := _claim(counters[0])) < wl.tags:
        for u in range(wl.units):
            x, private = wl.unit(t, u)
            t0 = time.perf_counter()
            unit, _, _, _ = fluxcode.encode_unit(x, p)
            enc_t += time.perf_counter() - t0
            nbytes += len(unit)
            nan_units += private
        with counters[2].get_lock():
            counters[2].value += 1
    switch.wait()
    while (t := _claim(counters[1])) < wl.tags:
        for u in range(wl.units):
            e = wl.encoded(t, u)
            t0 = time.perf_counter()
            fluxcode.decode_unit(e)
            dec_t += time.perf_counter() - t0
        with counters[3].get_lock():
            counters[3].value += 1
    out.put((enc_t, dec_t, nbytes, nan_units))


def _process_main(a, ready, go, switch, counters, out):
    wl = Workload(a)
    wl.build_pool()
    _warm(wl)
    ready.put(True)
    _work(wl, go, switch, counters, out)


def _warm(wl):
    for xs, encs in zip(wl.pool, wl.enc):
        fluxcode.decode_unit(encs[0])
        x = xs[0].copy()
        x[123:4567] = np.nan
        fluxcode.decode_unit(fluxcode.encode_unit(x, wl.params).unit)


def _watch(counter, total, t0, bins, interval=0.01):
    """Poll a per-tag counter until it reaches total; bins gets (elapsed s, tags done)."""
    while True:
        v = counter.value
        now = time.perf_counter()
        bins.append((now - t0, v))
        if v >= total:
            return now
        time.sleep(interval)


def run(a, wl):
    W = a.workers
    if a.mode == "threads":
        counters = [mp.Value("q", 0) for _ in range(4)]  # claimed / done, per phase
        go, switch = threading.Barrier(W + 1), threading.Barrier(W + 1)
        import queue
        out = queue.Queue()
        procs = [threading.Thread(target=_work, args=(wl, go, switch, counters, out)) for _ in range(W)]
    else:
        ctx = mp.get_context("spawn")
        counters = [ctx.Value("q", 0) for _ in range(4)]
        go, switch, out, ready = ctx.Barrier(W + 1), ctx.Barrier(W + 1), ctx.Queue(), ctx.Queue()
        procs = [ctx.Process(target=_process_main, args=(a, ready, go, switch, counters, out))
                 for _ in range(W)]
    for p in procs:
        p.start()
    if a.mode == "processes":
        for _ in range(W):
            ready.get()
    enc_bins, dec_bins = [], []
    go.wait()
    t0 = time.perf_counter()
    t1 = _watch(counters[2], wl.tags, t0, enc_bins)
    switch.wait()
    t1b = time.perf_counter()
    t2 = _watch(counters[3], wl.tags, t1b, dec_bins)
    res = [out.get() for _ in range(W)]
    for p in procs:
        p.join()
    return t1 - t0, t2 - t1b, res, enc_bins, dec_bins


# --- single-thread stage breakdown of one unit (where the time goes) ---

def breakdown(wl, reps=200):
    p = wl.params
    rows = []
    for k, xs in zip(wl.kinds, wl.pool):
        x = xs[1]
        X = x.reshape(BLOCKS, BLOCK)
        cz, dz = _api._zstd()
        head, param, anchor, resid, codes, _, _, _ = _api._encode_blocks(X, p)
        raw = _format.write_unit(head, param, anchor, resid, codes).data  # as _api._compress passes it
        frame = cz.compress(raw)
        unit = _format.pack_header(BLOCK, x.size) + frame
        out = np.empty((BLOCKS, BLOCK))

        def t(f, *args):
            best = np.inf
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(reps):
                    f(*args)
                best = min(best, (time.perf_counter() - t0) / reps)
            return 1e6 * best
        rawd = np.frombuffer(dz.decompress(frame), np.uint8)
        rows.append((k, len(unit) * 8 / x.size,
                     t(_api._encode_blocks, X, p), t(_format.write_unit, head, param, anchor, resid, codes),
                     t(cz.compress, raw), t(fluxcode.encode_unit, x, p),
                     t(dz.decompress, frame), t(_decoder.decode_unit, rawd, out),
                     t(fluxcode.decode_unit, unit)))
    return rows


def machine():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.model"], capture_output=True,
                             text=True, check=False).stdout.split()
        cpu = " ".join(cpu)
    except OSError:
        cpu = platform.processor()
    return f"{cpu}, {platform.system()} {platform.release()}, Python {platform.python_version()}, numpy {np.__version__}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="sensor-mix", help="preset (sensor-mix, all) or kind[:weight],... (see --list)")
    ap.add_argument("--tags", type=int, default=1000)
    ap.add_argument("--units", type=int, default=DAY_UNITS, help="1-minute units per tag (1440 = a day)")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of --tags to run (the day stays whole)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", choices=["threads", "processes"], default="threads")
    ap.add_argument("--pool", type=int, default=32, help="distinct units per signal kind")
    ap.add_argument("--nan-minutes", type=float, default=3.0, help="NaN minutes per tag per day")
    ap.add_argument("--nan-runs", type=int, default=2, help="runs those minutes are split into")
    ap.add_argument("--sprinkle", type=float, default=0.0, help="fraction of samples set to isolated NaN, every unit")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--breakdown", action="store_true", help="also print a single-thread per-stage breakdown")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("kinds:", ", ".join(ALL_KINDS))
        print("presets:", ", ".join(f"{k} = {v}" for k, v in PRESETS.items() if k != "all"), "; all = every kind")
        return
    a.tags = max(1, round(a.tags * a.scale))

    wl = Workload(a)
    t0 = time.perf_counter()
    wl.build_pool()
    worst = check_pool(wl)
    _warm(wl)
    gen_s = time.perf_counter() - t0

    n_units = wl.tags * wl.units
    samples = n_units * BLOCKS * BLOCK
    gb = 8 * samples / 1e9
    n_nan = len(wl.nan_spans)
    nan_samples = sum(e - s for sp in wl.nan_spans.values() for s, e in sp)
    print(f"# fluxcode day-scale stress test\n\n{machine()}\n")
    print(f"data `{a.data}`: {wl.tags} tags x {wl.units} units x {BLOCKS} x {BLOCK} = {samples / 1e9:.2f}e9 samples "
          f"({gb:.1f} GB float64); {a.workers} {a.mode}; pool {a.pool} units/kind")
    print(f"NaN: {a.nan_minutes:g} min/tag/day in {a.nan_runs} runs -> {n_nan:,} units touched "
          f"({100 * n_nan / n_units:.2f}%), {100 * nan_samples / samples:.3f}% of samples"
          + (f"; sprinkle {a.sprinkle:g} in every unit" if a.sprinkle else ""))
    print("tags per kind: " + ", ".join(f"{k} {int((wl.tag_kind == i).sum())}" for i, k in enumerate(wl.kinds)))
    print(f"(pool generated, roundtrip-checked (worst error {worst:.2e} of block range) and warmed in {gen_s:.0f} s)\n")

    if a.breakdown:
        print("## Single thread, one unit, µs (best of 5 x 200)\n")
        print("| kind | bits/sample | kernels | write_unit | zstd | encode_unit | unzstd | decode kernel | decode_unit |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in breakdown(wl):
            print(f"| {r[0]} | {r[1]:.2f} | " + " | ".join(f"{v:.0f}" for v in r[2:]) + " |")
        print()

    if a.mode == "processes":
        wl.pool, wl.enc = [], []  # the workers build their own
    te, td, res, eb, db = run(a, wl)
    enc_cpu = sum(r[0] for r in res)
    dec_cpu = sum(r[1] for r in res)
    nbytes = sum(r[2] for r in res)

    def line(name, wall, cpu):
        return (f"| {name} | {wall:.1f} s | {gb / wall:.2f} GB/s | {samples / wall / 1e6:.0f} M | "
                f"{1e6 * wall * a.workers / (n_units * BLOCKS):.2f} | {100 * cpu / (wall * a.workers):.0f}% |")
    print("## Result\n")
    print("| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |")
    print("|---|---|---|---|---|---|")
    print(line("encode", te, enc_cpu))
    print(line("decode", td, dec_cpu))
    print(f"\ncompressed {nbytes / 1e9:.2f} GB ({8 * nbytes / samples:.2f} bits/sample, ratio {gb * 1e9 / nbytes:.1f}); "
          f"real time is {wl.units * 60:,} s per day, so encode runs at {wl.units * 60 / te:.0f}x real time "
          f"for {wl.tags} tags.\n")

    def timeline(bins, name):
        """throughput per ~10% of the phase, to show thermal throttling on a sustained run."""
        if len(bins) < 3:
            return
        per_tag = wl.units * BLOCKS * BLOCK * 8 / 1e9
        T = bins[-1][0]
        edges = np.linspace(0, T, 11)
        ts = np.array([b[0] for b in bins])
        vs = np.array([b[1] for b in bins], float)
        v = np.interp(edges, ts, vs)
        rates = np.diff(v) * per_tag / np.diff(edges)
        print(f"{name} GB/s per tenth of the phase: " + " ".join(f"{r:.1f}" for r in rates))
    timeline(eb, "encode")
    timeline(db, "decode")


if __name__ == "__main__":
    main()
