"""Slow tags in 4-minute blocks (240 samples): bits/sample with the current NOISE_MIN_LEN."""
import sys, numpy as np
import corpus, harness as H
import fluxcode
from fluxcode import _encoder
tot = {}
for name in corpus.slow_names():
    x = corpus.slow(name)
    n = sum(len(fluxcode.encode_group(part, block_len=240).group) for part, _ in H.groups_of(x, 240, 60))
    tot[name] = 8 * n / len(x)
print("NOISE_MIN_LEN", _encoder.NOISE_MIN_LEN, " ".join(f"{k}:{v:.2f}" for k, v in tot.items()))
print("mean", np.mean(list(tot.values())))
