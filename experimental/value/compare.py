"""Compare two or more sizes.py JSONs: bits/sample per signal and per set (geometric mean of ratios)."""
import json, sys
import numpy as np
runs = [json.load(open(p)) for p in sys.argv[1:]]
labels = [p.rsplit("/", 1)[-1].removesuffix(".json") for p in sys.argv[1:]]
base = runs[0]
print("| set | signal | " + " | ".join(labels) + " | rmse " + " / ".join(labels) + " |")
print("|---|---|" + "---|" * len(labels) + "---|")
ratios = {}
for i, r in enumerate(base):
    cells = []
    for j, run in enumerate(runs):
        o = run[i]
        bs = 8 * o["bytes"] / o["n"]
        cells.append(f"{bs:.3f}" if j == 0 else f"{bs:.3f} ({100 * (o['bytes'] / r['bytes'] - 1):+.0f}%)")
        ratios.setdefault((r["set"], j), []).append(o["bytes"] / r["bytes"])
    rm = " / ".join(f"{run[i]['rmse']:.3g}" for run in runs)
    print(f"| {r['set']} | {r['name']} | " + " | ".join(cells) + f" | {rm} |")
for s in dict.fromkeys(r["set"] for r in base):
    print(s, " ".join(f"{labels[j]}: {100 * (np.exp(np.mean(np.log(ratios[(s, j)]))) - 1):+.1f}%" for j in range(len(runs))))
