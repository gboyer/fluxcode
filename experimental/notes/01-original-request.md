*Original request that started the project, lightly edited.*

I want to try some different "codecs" for linear time series data.

Programming language: Python

Desired Fidelity: This is just a toy test.  Some boundary condition decisions,
you can make arbitrarily; for example, if you're quantizing to something like
[0,255] there may be a question about numerical stability of 255 as a divisor
that we'll mostly ignore now. Make progress with the most reasonable decisions.

Your Judgment: Add more codecs and synthetic data sets if you think they are
interesting.

The goal of this codec is to compress 1000 samples (millisecond) within a
second into fewer bytes, and understand the loss.

# Codecs to test

Here are the options I want to test.

## 8 bit quantized

Store max and min for the region of 1000 samples as float64. Then
quantize each sample to [0,255].

## Binary Tree

This tries to exploit correlation across time and values hierarchically.

First, store the min/max as float64.

Assume a binary tree of 1024 samples, where the last 24 are omitted from the
tree (once the depth is sufficient)

Divide the space hierarchically in half. For each half, you store 2 bits,
the values meaning:

- 00: no narrowing, multiplier [0, 1] on min/max
- 01: narrow bidirectionally [1/4, 3/4]
- 10: narrow [0, 1/2]
- 11: narrow [1/2, 1]

The number of "internal nodes" should be 1001: 1 + 2 + 4 + 8 + 16 + 32 + 63 + 125 + 250 + 500

Each leaf is a simple 0 (bottom of region) or 1 (top of region).

There should be exactly 1000 leaves.

Note, when reporting stats on this, also include the frequency of the four binary tree codes.

## Linear piecewise, simple min/max

This assumes data is locally linear, but attempts to preserve min and max to keep spikes.

Algorithm:

* Divide the 1000 samples into groups of 16
* For each group of 16, find the min and max of that region
* Optional: Consolidate any min/max pairs at the boundaries to simplify monotonic series. (If the end of one block is a max and the beginning of the next block is a min, keep only the first point)

The encoding is:

* max: float64, min: float64 (like before)
* for each linear piecewise
  * 4 bits offset for the x axis (e.g. 0000 means the very next sample, 1111 means 16 samples ahead)
  * 12 bits where 0 is the min and 4095 is the max, linearly interpolated

### Variant: Cubic

Use standard cubic interpolation (I believe it's called PCHIP?)

### Variant: industrial-historian bounded-error piecewise linear (swinging door)

Use the standard industrial algorithm for linear piecewise. One thing to verify is if
short impulse peaks end up on the right sample number; maybe there are small variants.

## Meta Gorilla?

Meta has an exact algorithm that's based on xor, I think it's called Gorilla or similar.

# Data to test against

* Straight up linear from 0-999.  This presents a challenge for the Binary Tree
  codec because it's intentionally misaligned (last 24 samples)
* Simple quadratic
* Sinusoidal at 4Hz, 10Hz, and 50Hz
* Three Gaussian "spikes" with light noise the rest, at t=100ms, t=250ms, and
  t=500ms.  White noise has an amplitude of +/-10, Gaussian spikes have a
  standard deviation of 10 samples wide and peak of 1000 (in addition to the
  background noise)
* Maybe a couple other examples
* Report bytes of the final encoding, and compression ratio against the naive
  float64 per sample encoding

For each example report:

* Standard error of each sample (sqrt mean squared error)
* Error of the mean across the region
* Error of the max across the region
* Error of the min across the region
* A graph of each (html page, with images)

All error numbers are relative to (max-min): essentially "full scale error"
