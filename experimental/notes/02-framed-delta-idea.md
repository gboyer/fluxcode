*The framed-delta idea behind the `cw*-delta1-bfp` formats, lightly edited.*

I know I'm spitting out silly ideas, just want to get more examples of speed and performance
tradeoffs.

My next suggestion is still delta encoding in quantized integers, but divided into frames
with a base and magnitude.

At the beginning of the block you have a 2-bit exponent for each frame. If every sample in
a frame has a small delta, that exponent gives additional precision.

So if the exponent is 0, it is scaled as normal.  If the exponent is 3, then
each value moves only a quarter as much for that frame.
