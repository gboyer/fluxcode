# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Thread scaling of python-zstandard compressobj.compress with and without flush(BLOCK) (same call count):
flush holds the GIL (the compression of the buffered data happens inside it), compress alone releases it."""

import sys, time, threading, numpy as np, zstandard
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,"tests"); sys.path.insert(0,"bench")
from rust_decoder import best
rng=np.random.default_rng(0)
# 16 KB chunks of compressible data, many calls per task
chunk=(rng.integers(0,8,16384,dtype=np.uint8)).tobytes()
N=400
def mk(flush, calls_split=1):
    def task(_):
        c=zstandard.ZstdCompressor(level=3).compressobj()
        out=0
        for _ in range(N):
            out+=len(c.compress(chunk))
            if flush: out+=len(c.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK))
        out+=len(c.flush()); return out
    return task
mib=N*16384*16/2**20
for name,fn in [("compress only",mk(False)),("compress + flush(BLOCK)",mk(True))]:
    for si in (0.005,0.00001):
        sys.setswitchinterval(si)
        row=[]
        for nt in (1,4,8):
            with ThreadPoolExecutor(nt) as ex:
                run=lambda: list(ex.map(fn,range(16))); run()
                row.append(mib/best(run,3))
        print(f"{name:26s} switchinterval {si:g}: "+", ".join(f"{nt}t {r:.0f}" for nt,r in zip((1,4,8),row))+" MiB/s")
