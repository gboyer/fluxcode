"""Encode and decode time with the period predictor off (effort 5) and on (effort 6), the only
difference between those efforts. Interleaved in one process, best of 7, per corpus signal."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np
import corpus, harness as H  # noqa: F401
import fluxcode

def best(fn, reps=7):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)

rows = []
for s, name, x, bl, bpg in corpus.corpus(sys.argv[1:] or ["fast", "slow", "synth"]):
    parts = [p for p, _ in H.groups_of(x, bl, bpg)]
    groups = {}
    for effort in (5, 6):
        groups[effort] = [fluxcode.encode_group(p, fluxcode.Params(effort=effort), block_len=bl).group for p in parts]
        for g in groups[effort]: fluxcode.decode_group(g)
    used = sum(int((np.frombuffer(__import__("fluxcode")._group.decompress(g).raw_body, np.uint8)[:__import__("fluxcode")._group.decompress(g).header.num_blocks] & 0x80 != 0).sum()) for g in groups[6])
    nblocks = sum(fluxcode._group.decompress(g).header.num_blocks for g in groups[6])
    enc, dec = {}, {}
    for _ in range(2):  # alternate the order, keep the best of both rounds
        for effort in (5, 6):
            e = best(lambda: [fluxcode.encode_group(p, fluxcode.Params(effort=effort), block_len=bl) for p in parts])
            d = best(lambda: [fluxcode.decode_group(g) for g in groups[effort]])
            enc[effort] = min(enc.get(effort, 1e9), e); dec[effort] = min(dec.get(effort, 1e9), d)
    rows.append((s, name, used, nblocks, enc, dec, len(x)))
    print(f"{s:5s} {name:24s} period blocks {used:3d}/{nblocks:3d}  encode {enc[5]/nblocks*1e6:6.2f} -> {enc[6]/nblocks*1e6:6.2f} us/block ({enc[6]/enc[5]-1:+5.0%})"
          f"  decode {dec[5]/nblocks*1e6:5.2f} -> {dec[6]/nblocks*1e6:5.2f} ({dec[6]/dec[5]-1:+5.0%})", flush=True)

for label, sel in (("signals using it", lambda r: r[2] > 0), ("signals not using it", lambda r: r[2] == 0), ("all", lambda r: True)):
    sub = [r for r in rows if sel(r)]
    if not sub: continue
    e5, e6 = sum(r[4][5] for r in sub), sum(r[4][6] for r in sub)
    d5, d6 = sum(r[5][5] for r in sub), sum(r[5][6] for r in sub)
    print(f"{label:22s} ({len(sub)} signals): encode {e6/e5-1:+.1%}, decode {d6/d5-1:+.1%}")
