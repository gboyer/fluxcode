"""Prototype of escaped samples (no format yet): per block, a free trigger (the mean within 5% of the range
from one end), the bulk from a trimmed mean and mean deviation, candidates beyond the bulk, and the benefit
rule (at most n/16 escapes; the bulk's 16-bit step at least 4x finer; the noise floor not already coarser).
Escaped samples are held (previous kept value) for encoding; payloads charged at 3 bytes each (sign, exponent,
~16 mantissa bits at the block's old step), zstd ignored. Reports glitch accuracy and false triggers."""
import math, sys
import numpy as np
import corpus, harness as H
import fluxcode
from fluxcode import _encoder, _noise

def escapes_for(block, f=0.25):
    n = len(block); lo, hi = block.min(), block.max(); rng = hi - lo
    if n < 16 or rng <= 0:
        return np.zeros(n, bool)
    m = block.mean()
    if not (m - lo < 0.05 * rng or hi - m < 0.05 * rng):
        return np.zeros(n, bool)
    # the bulk: trimmed mean and mean deviation (twice), then everything beyond 8 deviations of it
    c = np.median(block[::4]) if False else m
    for _ in range(3):
        d = np.abs(block - c); s = d.mean()
        inside = d <= 4 * 1.2533 * s
        c = block[inside].mean()
    d = np.abs(block - c); s = 1.2533 * d[d <= 4 * 1.2533 * d.mean()].mean()
    cand = d > max(16 * s, 1e-300)
    k = cand.sum()
    if k == 0 or k > n // 16:
        return np.zeros(n, bool)
    bulk = block[~cand]
    old_e = _encoder.range_exponent(lo, hi, 16)
    # Already exact at the full range (a decimal grid within 16 bits, as detection would find): nothing to gain
    for places in range(0, 7):
        if np.array_equal(np.round(block, places), block) and rng * 10 ** places < 2 ** 15:
            return np.zeros(n, bool)
    new_e = _encoder.range_exponent(bulk.min(), bulk.max(), 16)
    sigma, rho = _noise.noise(bulk, _encoder.range_scale(bulk.min(), bulk.max()), np.empty(n))
    floor_e = math.floor(math.log2(f * sigma)) if rho > _encoder.NOISE_GATE and sigma > 0 else -10**9
    gain = old_e - max(new_e, floor_e)
    return cand if gain >= 2 else np.zeros(n, bool)

def encode_with_escapes(x, bl=1000):
    held = x.copy(); n_esc = 0; mask = np.zeros(len(x), bool)
    for b in range(0, len(x), bl):
        e = escapes_for(x[b:b + bl])
        idx = np.nonzero(e)[0] + b
        for i in idx:
            held[i] = held[i - 1] if i > b else x[b:b + bl][~e][0]
        mask[idx] = True; n_esc += len(idx)
    groups = [fluxcode.encode_group(held[s:s + 60000], block_len=bl).group for s in range(0, len(x), 60000)]
    y = np.concatenate([fluxcode.decode_group(g).values for g in groups])
    y[mask] = x[mask]  # exact (as the payload would round them to the old step: tiny)
    return sum(map(len, groups)) + 3 * n_esc, y, n_esc

if __name__ == "__main__":
    rng = np.random.default_rng(8); n = 240_000
    x = 5 + 0.3 * np.sin(np.arange(n) / 3000) + rng.normal(0, 0.002, n)
    glitch = rng.choice(n, 40, replace=False)
    for code in (-9999.0, 1e6, 0.0):
        y = x.copy(); y[glitch] = code
        base = [fluxcode.encode_group(y[s:s + 60000]).group for s in range(0, n, 60000)]
        d0 = np.concatenate([fluxcode.decode_group(g).values for g in base])
        size, d1, k = encode_with_escapes(y)
        e0 = np.abs(d0 - y); e1 = np.abs(d1 - y)
        hit = np.unique(glitch // 1000)
        print(f"glitch {code:>8}: bytes {sum(map(len, base))} -> {size} ({k} escaped); worst error in glitched blocks "
              f"{e0.reshape(-1, 1000)[hit].max():.3g} -> {e1.reshape(-1, 1000)[hit].max():.3g}")
    # false triggers on the corpus: blocks with escapes, and size change
    for set_name, name, xs, bl, bpg in corpus.corpus(["fast", "slow", "synth"]):
        size, _, k = encode_with_escapes(xs, bl)
        if k:
            base = sum(len(fluxcode.encode_group(p, block_len=bl).group) for p, _ in H.groups_of(xs, bl, bpg))
            print(f"  {set_name} {name}: {k} escaped, size {100*(size/base-1):+.1f}%")
    print("corpus done")
