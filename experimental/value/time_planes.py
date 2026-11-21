"""Time residuals of historian archives: bit planes (today) vs byte planes, zstd 3, just that field."""
import numpy as np
import zstandard
import corpus  # noqa: F401 (paths)
import harness as H
from fluxcode import _group
from tslab.common import historian as hist

def bit_planes(u):  # u: uint64 residuals (< 2^32 here), padded to octets
    n = -(-len(u) // 8) * 8
    v = np.zeros(n, np.uint64); v[:len(u)] = u
    bits = ((v[None, :] >> np.arange(32, dtype=np.uint64)[:, None]) & np.uint64(1)).astype(np.uint8)
    return np.packbits(bits.reshape(32, -1, 8)[:, :, ::-1], axis=2).tobytes()

def byte_planes(u):
    v = u.astype(np.uint32)
    return np.stack([(v >> (8 * j)) & 0xFF for j in range(4)]).astype(np.uint8).tobytes()

z = zstandard.ZstdCompressor(level=3)
print("| tag | points | H bits | bit planes bits/pt | byte planes bits/pt |"); print("|---|---|---|---|---|")
for tag in hist.TAGS:
    tb = tbit = tbyte = h = 0; pts = 0
    for day in range(3):
        x, idx = hist.tag_day(tag, day)
        t = (hist.DAY_START_NS + idx.astype(np.int64) * hist.SECOND_NS)
        # hourly blocks
        b = idx // 3600
        sizes = np.bincount(b, minlength=24).astype(np.int64)
        flags = np.zeros(24, np.uint8)
        rows = _group.encode_time_rows(t, sizes, flags)
        irregular = (flags & 16) != 0
        offs = np.r_[0, np.cumsum(sizes)]
        u = np.concatenate([rows.residuals[offs[k]:offs[k + 1]] for k in range(24) if irregular[k]] or [np.zeros(0, np.uint64)])
        tbit += len(z.compress(bit_planes(u))); tbyte += len(z.compress(byte_planes(u)))
        h += H.entropy_bits(u); pts += len(u)
    print(f"| {tag} | {pts // 3} | {h / pts:.2f} | {8 * tbit / pts:.2f} | {8 * tbyte / pts:.2f} |")
