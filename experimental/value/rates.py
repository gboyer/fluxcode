"""One physical process sampled at different rates: what does the floor do?
Process: setpoint steps through a first-order lag (tau 60 s) + OU disturbance (tau 20 s, sd 0.5) + loop hunting
(0.2 Hz, 0.3) + sensor white noise sd 0.05; point-sampled. 20 blocks of 1000 samples per rate."""
import sys
import numpy as np
import harness as H  # noqa: F401
import fluxcode

def process(dt, n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    # setpoint: a step every ~30 min on average, through a 60 s lag (exact response at sample times)
    sp = np.zeros(n)
    k, level, prev = 0.0, 50.0, 50.0
    steps = np.cumsum(rng.exponential(1800, int(n * dt / 1800) + 5))
    lagged = np.full(n, 50.0)
    for s in steps:
        if s > t[-1]:
            break
        new = rng.uniform(40, 60)
        lagged += np.where(t >= s, (new - level) * (1 - np.exp(-(t - s) / 60)), 0)
        level = new
    a = np.exp(-dt / 20)
    e = rng.normal(0, 0.5 * np.sqrt(1 - a * a), n)
    from scipy.signal import lfilter
    ou = lfilter([1], [1, -a], e)
    hunt = 0.3 * np.sin(2 * np.pi * 0.2 * t + 1)
    return lagged + ou + hunt + rng.normal(0, 0.05, n), lagged + ou + hunt

print(f"k = {__import__('os').environ.get('FLUXCODE_X_NOISE_ORDER', '2')}")
print("| rate | bits/sample | RMSE / sensor σ | RMSE / process step σ | max err / block range |")
print("|---|---|---|---|---|")
for label, dt in [("1 kHz", 0.001), ("100 Hz", 0.01), ("10 Hz", 0.1), ("1 Hz", 1.0), ("0.1 Hz", 10.0), ("1/min", 60.0)]:
    x, clean = process(dt, 20_000)
    g = fluxcode.encode_group(x, block_len=1000).group
    y = fluxcode.decode_group(g).values
    err = y - x
    innov = np.std(np.diff(clean))  # the process's own move per sample
    worst = max(np.abs(err[b:b + 1000]).max() / np.ptp(x[b:b + 1000]) for b in range(0, 20_000, 1000))
    print(f"| {label} | {8 * len(g) / len(x):.2f} | {np.sqrt(np.mean(err ** 2)) / 0.05:.3f} | "
          f"{np.sqrt(np.mean(err ** 2)) / innov:.4f} | {100 * worst:.4f}% |")
