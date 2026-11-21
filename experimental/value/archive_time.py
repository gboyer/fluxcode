"""Time vs value cost on simulated historian archives (swinging door at each tag's CompDev): a day per tag, hourly
time blocks in one day group."""
import numpy as np
import corpus  # noqa: F401  (sys.path)
import harness as H  # noqa: F401
import fluxcode
from tslab.common import historian as hist
from tslab.prune import signals

print("| tag | points/day | values only B | values+times B | time share | B/point times |")
print("|---|---|---|---|---|---|")
for tag in hist.TAGS:
    tot_v = tot_t = pts = 0
    tot_h = 0
    for day in range(3):
        x, idx = hist.tag_day(tag, day)
        v = x[idx]
        t = (hist.DAY_START_NS + idx.astype(np.int64) * hist.SECOND_NS).astype("datetime64[ns]")
        g1, *_ = fluxcode.encode_time_blocks(v, t, start_time=np.datetime64(hist.DAY_START_NS, "ns"),
                                             block_duration=np.timedelta64(3600, "s"))
        # values alone, same block sizes
        sizes = fluxcode.decode_group(g1).block_sizes
        gc, *_ = fluxcode.encode_time_blocks(np.zeros(len(v)), t, start_time=np.datetime64(hist.DAY_START_NS, "ns"),
                                             block_duration=np.timedelta64(3600, "s"))
        gz = fluxcode.encode_blocks(np.zeros(len(v)), sizes).group
        tot_v += len(g1) - (len(gc) - len(gz)); tot_t += len(g1); pts += len(v)
        Hg = H.entropy_bits(np.diff(idx))
        tot_h = locals().get("tot_h", 0) + Hg
    print(f"| {tag} | {pts // 3} | {tot_v // 3} | {tot_t // 3} | {100 * (tot_t - tot_v) / tot_t:.0f}% | {8 * (tot_t - tot_v) / pts:.1f} bits (H {tot_h / pts:.1f}) |")
