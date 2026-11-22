# Model provenance and licence

## Origin

Both files in this directory are unmodified copies of the tinyllamas 260K
model and tokenizer released by Andrej Karpathy for the `llama2.c` project.

| Item | Source |
|---|---|
| Model repository | <https://huggingface.co/karpathy/tinyllamas> |
| Model card, 260K | <https://huggingface.co/karpathy/tinyllamas/blob/main/stories260K/readme.md> |
| Code and MIT licence | <https://github.com/karpathy/llama2.c> |
| Training data | TinyStories, not redistributed here |

| File | Bytes | SHA-256 |
|---|---:|---|
| `stories260K.bin` | 1,056,540 | `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696` |
| `tok512.bin` | 6,227 | `037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312` |

The hashes were checked on 2026-11-22 against the Hugging Face
`x-linked-etag` headers for the two files, and against a fresh download of
each file. They match byte for byte. The upstream repository was last
modified on 2023-08-15.

## Licence

The tinyllamas model card declares `license: mit`, and the Hugging Face model
metadata carries the `license:mit` tag. The `llama2.c` repository is also MIT
licensed. The upstream MIT notice is reproduced in
[`model/LICENSE`](LICENSE).

MIT permits use, copy, modification, merge, publication, distribution,
sublicensing, and sale, provided that the copyright notice and permission
notice are included in all copies or substantial portions. This repository
includes that notice in `model/LICENSE` and credits the author below.

The model and tokenizer files remain under MIT. They are not relicensed as
GPL-3.0-only. MIT is a GPL-compatible licence, so distributing these MIT
files alongside the GPL-3.0-only firmware and tools is permitted. The GPL
covers the firmware and analysis code, not `model/`.

## Credit

- Model and tokenizer: Andrej Karpathy, `tinyllamas` and `llama2.c`.
- Training data: the TinyStories dataset by Ronen Eldan and Yuanzhi Li. The
  dataset is not included in this repository.

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

## Obtaining the files independently

The files are included so that the paper's logit-level accuracy comparisons can
be reproduced against the exact checkpoint. To fetch them from upstream
instead:

```bash
curl -L -o stories260K.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
curl -L -o tok512.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin
sha256sum stories260K.bin tok512.bin
```

The two hashes must match the table above. No other revision or checkpoint is
covered by the paper's measurements.
