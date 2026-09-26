*The power-of-two proposal that became the fluxcode prototype (`fluxproto`), lightly edited.*

Real data tends to be discretized, although we don't always know at what level.

Here's my suggested encoding that takes lessons of the one-shot encoder (`delta0123-zstd`) but makes them a bit tighter.

- Quantization: Always choose a power of 2 range.
  - Store min as float64, as well as max-min. The latter is primarily for summary statistics.
  - Use the exponent of max-min to determine the quantization.
  e.g. if value-min is `1.011010011111011b*2^-5`, then it will quantize to `1011010011111011b`
  in a 16-bit quantization. The min of the block will be represented exactly, but the max will
  be close.
    - This approach optimizes for numerical stability of edits within a block are needed; it
    avoids repeated rounding and quantization errors, but does sometimes squeeze out dynamic
    range if an intermediate edit has an outlier.
  - A parameter will allow quantizations between [9,16] bits, but 16 bits will be the default
  (when testing, do a sweep). The parameter is implemented by a simple shift-right.
  The parameter will be assumed fixed within a test run: no need to do the final step of
  determining B to fit within 8 bits per sample; this helps with numerical stability and is
  unlikely to trigger very much on real data.
- Delta codec:
  - With the quantized value, test orders 0, 1, 2, 3.
    - Profile this step specifically; if it dominates, 1 is a reasonable default for sensors
    (2 and 3 rule for exact continuous only, 0 wins for relatively rare scenarios)
- Pre-compressed Encoding:
  - Per block:
    - Store the order.
    - Store the signed exponent. The exponent encompasses both the dynamic
    range of max-min as well as the quantization, so that `quantized * 2^exponent` is the
    reconstructed value (after the delta code)
    - Store the low order bytes of the quantized deltas
    - Store the high order bytes
  - All blocks are interleaved, organized by type of data:
  `(order x N, exponent low x N, exponent high x N, low x N, high x N)`.
  I expect the latter to compress better than one block at a time.
