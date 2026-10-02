# fluxcode research: lossy codecs for 1 kHz time series

This directory is the investigation that led to [fluxcode](../README.md): dozens of codec variants
for float64 time series in 1000-sample blocks, the benchmarks that compared them, and the report
that explains the design. **The findings are in [REPORT.md](REPORT.md).** The original request and
the design prompts are in [notes/](notes/).

- **Synthetic data only.** Every signal is generated (`tslab/common/datasets.py`); nothing here has
  been validated on real data yet.
- **Not maintained and not a library.** The code is kept so the report can be reproduced. It isn't
  covered by CI, and its APIs may change or break without notice. Use the `fluxcode` package for
  anything real.
- **License:** MIT, like the rest of the repository ([LICENSE](../LICENSE)).

## Setup

```sh
cd experimental
uv sync          # Python 3.11 (pinned in .python-version); fluxcode comes from .. as an editable path dependency
```

## The codecs

Every codec is measured on one-minute units of 60 blocks of 1000 samples, and charged every byte
its decoder needs (`tslab/common/unit.py`). Names list the stages in processing order, parameters
last: `[cw<bits>-|quant<bits>-|ratectl-]<predictor>-<coder or step mapping>[-<variant>][-<B>]`.
`delta1` is a fixed first difference; `delta0123` picks the order per block from {0, 1, 2, 3};
`cw<bits>` is constant width; `bfp` is block floating point.

| module | codecs | what they are |
|---|---|---|
| `tslab/common/datasets.py` | — | every test signal: 12 kinds as continuous one-minute signals, discretized variants, noisy sets, and the report's units |
| `tslab/common/unit.py` | — | the unit harness: `Concat` and `SharedBackend` turn block codecs into unit codecs |
| `tslab/common/bitio.py`, `intcode.py` | — | bit writer/reader; zigzag, varint, residuals and the order pick |
| `tslab/classic/quant.py` | `quant8`, `quant8-delta1-deflate`, `quant6-delta1-deflate` | uniform quantization over the block's range, with and without delta + deflate |
| `tslab/classic/dpcm.py` | `dpcm6-linear-deflate`, `-cubic-`, `-mulaw-`, `dpcm4-*-deflate` | closed-loop DPCM with companded step tables, then deflate |
| `tslab/classic/bintree.py` | `bintree-center`, `bintree-edge` | the 2-bit hierarchical narrowing tree from the original request |
| `tslab/classic/piecewise.py` | `pwlinear-minmax16`, `pwlinear-minmax16-merged`, `pchip-minmax16` | min/max knots per 16 samples, linear or PCHIP |
| `tslab/classic/swinging_door.py` | `swingdoor-0.5%`, `swingdoor-2%` | swinging-door trending |
| `tslab/classic/gorilla.py` | `gorilla-xor` | Gorilla's lossless XOR float encoding |
| `tslab/classic/dct.py` | `dct-top64` | keep the 64 largest DCT coefficients |
| `tslab/entropy/ratectl.py` | `delta0123-rice`, `delta0123-zstd`, `delta0123-deflate`, `flac`, `best: …` | quantize → predict → entropy code with a searched step (rate control) |
| `tslab/entropy/ratectl_codecs.py` | `ratectl-delta0123-zstd-4b`, `ratectl-flac-4b` | the rate-controlled pipelines as unit codecs for the matrix |
| `tslab/entropy/delta_zstd.py` | `delta0123-zstd-B`, `delta12-zstd-8`, `delta0123-deflate-B-L*` | the one-shot encoder: step from the block's range, order by variance, zstd per unit |
| `tslab/entropy/native.py` | `quant8`, `quant8-delta1-deflate`, `quant8-delta1-zstd`, DPCM | numba ports, byte-compatible with the Python references |
| `tslab/constwidth/delta_linear.py`, `delta_sqrt.py` | `cw8-delta1-linear`, `cw8-delta1-sqrt` | closed-loop delta1 at 8 bits per sample, uniform or √-scaled steps |
| `tslab/constwidth/delta_bfp.py` | `cw8-delta1-bfp-e2`, `cw8-delta1-bfp-e4`, `cw6-…`, `cw4-…` | closed-loop delta1 with a block-floating-point exponent per 16-sample frame |
| `tslab/constwidth/delta_tree.py` | `cw-delta1-tree-N` | delta1 with hierarchical block floating point over the deltas |
| `tslab/flux/proto.py` | `fluxproto-B[-nfF]` | the fluxcode prototype: power-of-two step, decimal detection, byte/bit/nibble planes, noise floor |
| `tslab/flux/adapters.py` | `fluxcode-B[-fF]` | the real fluxcode package as a unit codec, and its sweeps |

## The report

```sh
uv run python make_report.py        # report/index.html and its SVGs    (~3 min)
uv run python make_rate_report.py   # report/rate.html                   (~3 min)
uv run python make_time_report.py   # report/time.html                   (~5 s)
```

`make_report.py` runs every codec on the report's units (`tslab.common.datasets.units()`: 12 kinds ×
5 one-minute signals, 3,600 blocks), checks every decode against its error bound, and writes
`report/index.html` with these sections, in this order:

- a header (the data, the metrics, how to regenerate) and a contents list;
- a **codec legend**: one row per codec in a fixed order (family, then bit budget), with lossless /
  constant-width flags, a description and a module link;
- a **summary matrix**: codecs × datasets, bits/sample and RMSE per cell, shaded by RMSE;
- **size vs error**: one scatter of all codecs with a band for the lossless ones, then per-dataset panels;
- **per-dataset sections**: a table per dataset, and a three-panel SVG of one chosen second (the original,
  the decoded traces of representative codecs, their errors; zoomed where `KIND_INFO` defines a window);
- **parameter sweeps**: the fluxcode B sweep, the noise-floor sweep and the DPCM sweep;
- **design notes**.

Charts are SVG, and the build fails if a chart has no data. Each script deletes its own earlier outputs
in `report/` before writing. `make_rate_report.py` runs the rate-control experiment of REPORT.md §6 on
"regime" minutes and writes `report/rate.html` in the same style. `make_time_report.py` writes
`report/time.html`: only fluxcode's time axis (docs/SPEC.md §4), i.e. what exact timestamps cost for the
common clock shapes (a perfect grid, a grid with a few gaps, a noisy clock) and some harder ones; its
timings need a quiet machine. `report/` is generated, and committed so it can be read without rerunning.

## Benchmarks

Each answers one question from the report. Runtimes are on an Apple M3, one thread.

| script | question | runtime |
|---|---|---|
| `bench/native_speed.py` | How much faster are the numba ports than the Python references? (§8) | ~45 s |
| `bench/chunking.py` | What do one-minute chunks save over per-block compression? (§9) | ~11 s |
| `bench/delta_zstd.py` | How fast and how small is delta0123-zstd per minute, and what does 200 signals × 1 day cost? (§9) | ~7 s |
| `bench/constwidth_8bit.py` | Which constant-width ~8-bit format is best, and how does it compare with entropy coding? (§10) | ~8 s |
| `bench/fluxproto_sweep.py` | The prototype: B sweep, discretized data, decimal detection, byte/bit/nibble planes (§11) | ~22 s |
| `bench/fluxproto_noise_floor.py` | What does the noise floor save, and what error does it add? (§11) | ~2 s |
| `bench/fluxproto_truncation.py` | Block step vs zeroing low bits per sample; can spikes keep full precision? (§11) | ~1 s |
| `bench/fluxproto_plane_models.py` | Would per-plane probability models beat zstd? (§11) | ~2 s |
| `bench/fluxproto_raw_planes.py` | Should near-random planes bypass zstd? (§11) | ~1.5 s |
| `bench/fluxproto_relative_sign.py` | Does a relative-sign zigzag help? (§11) | ~1 s |
| `bench/fluxcode_sweep.py` | The same sweep on the fluxcode package (§11) | ~7 s |

Run one with `uv run python -m bench.<name>` from this directory.

[plane_coders/](plane_coders/README.md) is a separate experiment: would an RLE or arithmetic coder beat zstd on the
residual bit planes (it is faster, and larger; not adopted)?
[plane_layout/](plane_layout/README.md) asks whether bit planes vs byte planes can be predicted without compressing both;
what it found became `Params.effort`.
[rust_port/](rust_port/README.md) ports the encoder and decoder to Rust: only the zstd frame build is worth it (python-zstandard
holds the GIL in a block flush); it became the optional `fluxcode[rust]` extension.
