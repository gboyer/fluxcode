*The fluxcode API sketch; the implemented plan is [docs/design/api-plan.md](../../docs/design/api-plan.md). Lightly edited.*

I've been thinking more about getting the power-of-two codec (`fluxproto`) ready to ship.

Final state is to expose the steps separately, and together.

Let's pick the following code name: fluxcode

We will do a git init, and create a "fluxcode" package. Modules:

- `fluxcode.py`: the encoder/decoder entry point
  - `encode(data: arraytype, *, <params>): FluxEncode`
    - data: it should be arrow or numpy, whichever our underlying implementation supports
      faster. pandas series is ok too.
    - the caller can specify absolute min/max number of bits
      - `min_quantize_bits: int = 9`
      - `max_quantize_bits: int = 16`
    - the caller can trigger order detection
      - `diff_orders: frozenset[int]`. default is {0,1,2,3}. If the set has just one member, diff
      detection is off. Empty set is an error.
    - the caller can specify noise floor
      - `noise_floor_sigma: float = 0.25` (more conservative than previously)
      - note, B will be clamped within min/max quantize bits
    - the caller can also specify a max target bit rate
      - `target_bits_per_sample: int|None = None`
      - if this is set, use the entropy check and to estimate a cap on B
      - this is meant for workloads where a maximum bitrate is desired, but it's not a hard
      max.
    - note: B is determined with the following precedence
      - min/max_quantize_bits reigns supreme, never violated
      - target_bits_per_sample can constrain sigma as a max; if it's <= min, min is chosen
      - the noise floor paramter can constrain within that range
  - `decode(encoded_data: bytes)`: numpy/pyarrow type

Then helper modules. This is a basic sketch, mostly to illustrate the API shape:

  - `types.py`: defines core dataclass constructs like
    - QuantizeHeader: parameters that quantization needs stored in the final encoding:
      - min: float
      - range: float (this does not need to be stored; the decoder ignores it)
      - quantize_bits, anything for decimal decoding
    - QuantizeBlock: int16 array plus QuantizeHeader
    - DeltaHeader: anything the delta decoder needs, likely just order
    - DeltaBlock: int16 array plus DeltaHeader; likely this also requires the QuantizeHeader
  - `flux_quantize.py`: The quantizer:
    - `encode(data: arraytype, *, <params>): QuantizeBlock`
      - quantizes a single block only
    - `decode(block: QuantizeBlock): bytes`
  - `flux_delta.py`: the delta encoding and chooser
    - `encode(block: QuantizedBlock, *, <params>): DeltaBlock`
    - `decode(block: DeltaBlock): QuantizeBlock`
  - `flux_bitstream.py`: turns into actual bit streams
    - `encode(blocks: list[DeltaBlock]): bytes`
      - encodes the headers
      - does zigzag
      - encodes bit planes
      - calls zstd
    - `decode(data: bytes, block_min: float64_array): list[DeltaBlock]`
      - it assumes the caller is storing the float64 array of minima separately
  - `flux_entropy.py`: probably need a better name, but you need a module that the encoder uses
    to support noise floor and target_bits_per_sample.

One thing I'm fuzzy on: how much interdependency is there; how much is each phase aware of the
next.

- Does quantization happen before we determine the final B value, or after?
  Do we have to do multiple rounds?
