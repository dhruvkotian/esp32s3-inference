# Model provenance

Both files in this directory are unmodified copies of the tinyllamas 260K
model released by Andrej Karpathy alongside `llama2.c`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `stories260K.bin` | 1,056,540 | `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696` |
| `tok512.bin` | 6,227 | `037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312` |

Source:

- <https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin>
- <https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin>

Both hashes were checked against those downloads when this release was
prepared, and they match byte for byte.

## Model card

| Field | Value |
|---|---|
| Architecture | decoder-only transformer, llama2.c legacy v0 layout |
| Parameters | 260K |
| dim | 64 |
| layers | 5 |
| heads | 8 |
| KV heads | 4 |
| head dimension | 8 |
| vocabulary | 512 |
| context | 512 |
| training data | TinyStories |
| reported validation loss | 1.297 |

The file is the fp32 checkpoint. The firmware quantises the weight matrices to
int8 with per-row scales at load time; the checkpoint itself is not modified.

## Licence

`llama2.c` and the tinyllamas model files are released under the MIT licence.
The model weights are redistributed here under those terms. This is separate
from the GPL-3.0-only licence that covers the firmware in this repository.

## Why the weights are included

The paper's accuracy numbers are logit-level comparisons against this exact
checkpoint. Shipping the file removes the step where a reader downloads a
model, and removes the possibility that a different revision produces
different logits. The hash is the identity check.
