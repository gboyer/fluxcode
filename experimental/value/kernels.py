import math, time
import numpy as np
import harness as H
from fluxcode import _encoder, _noise
import sys; sys.path.insert(0, str(H.ROOT / "tests"))
from _signals import minute
def best(f, reps=2000):
    f(); b = 1e9
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(reps): f()
        b = min(b, (time.perf_counter() - t) / reps)
    return b * 1e6
for kind in ("sin-4.12hz", "noisy-sine"):
    x = minute(kind, 1)[:1000].copy()
    lo, hi = x.min(), x.max()
    e = _encoder.range_exponent(lo, hi, 16)
    q = np.empty(1000, np.int32); _encoder.quantize(x, lo, e, q)
    buf = np.empty(1000)
    sc = _encoder.range_scale(lo, hi)
    qq = q.copy()
    print(kind,
          f"pick {best(lambda: _encoder.pick_order(q, 15)):.2f}",
          f"noise {best(lambda: _noise.noise(x, sc, buf)):.2f}",
          f"center {best(lambda: (qq.__setitem__(slice(None), q), _encoder.center(qq, 0, 0, e, np.int64(0), math.ldexp(1.0, e) * 3, np.int64(0)))):.2f}",
          f"copy {best(lambda: qq.__setitem__(slice(None), q)):.2f}",
          "µs")
