# Release provenance: research tree to public export

This repository is a filtered export of a private research tree. The manuscript,
its LaTeX source, the built PDFs and the submission record were removed from
every commit before publication, and a small set of files was reworded so they
no longer reference those paths. Filtering rewrites commit hashes, so the public
tags are **new commits**, not the original research commits. This document
records the mapping and the verification behind it.

## Why the hashes differ

The public history was produced with `git filter-branch`:

- an index filter removed `paper/` and `submission/` from every commit;
- a tree filter reworded the files listed below and regenerated
  `MANIFEST.sha256` for the filtered tree;
- commit messages, authors and author dates were preserved.

The result is a history that contains no manuscript content in any commit, but
whose commit IDs necessarily differ from the research tree.

## Release mapping

| Release | Role | Research commit | Public commit | Public date |
|---|---|---|---|---|
| `v1.2.0` | Research baseline | `bb021fa` | `ea2cd1c` | 2026-11-22 |
| `v1.3.0` | Runtime evolution | `4fd16f0` | `e5b340b` | 2026-11-26 |
| `v1.3.1` | Metadata patch | none (public-only) | `6b23d04` | 2026-11-26 |

`v1.3.1` has no research-tree counterpart: it changes documentation and
metadata only, and its firmware is byte-identical to `v1.3.0`.

The research commit IDs are recorded for provenance only. They are not
reachable from this repository and must not be cited as public commits. The
public commit is the one to use with this repository.

`e5b340b` carries one additional public commit over the filtered form of
`4fd16f0`. That commit adds `CHANGELOG.md`, `CONTRIBUTING.md` and `SECURITY.md`,
rewrites the release README, and removes `tools/verify_submission_pdf.py`. It is
why `v1.3.0` has three public-only paths.

## Verification method

Content was compared by Git blob ID. A Git blob ID is the hash of the file
content, so an equal blob ID at the same path means byte-identical content.

```bash
git -C <research-tree> ls-tree -r v1.2.0
git -C <public-repo>   ls-tree -r v1.2.0
```

The same comparison was run for `v1.3.0`. Paths present on only one side were
classified separately.

## Verification results

| Release | Paths in common | Byte-identical | Intentionally changed | Research-only paths |
|---|---:|---:|---:|---:|
| `v1.2.0` | 128 | 123 | 5 | 14 |
| `v1.3.0` | 164 | 157 | 7 | 15 |

Identical paths by top-level directory:

| Directory | `v1.2.0` identical | `v1.3.0` identical |
|---|---:|---:|
| `firmware/` | 11 | 17 |
| `micro/` | 6 | 6 |
| `tools/` | 12 | 18 |
| `logs/` | 84 | 105 |
| `model/` | 4 | 4 |
| `config/` | 2 | 2 |
| `data/` | 1 | 1 |

Every path under `firmware/`, `micro/`, `tools/`, `logs/`, `model/`, `config/`
and `data/` that exists in both trees is byte-identical at both tags. This
includes the int8 kernels, the ESP-DSP microbenchmark, the raw UART captures,
the model and tokenizer, the sdkconfig snapshots and the reference logits.

## What was changed

Intentionally changed paths:

| Path | Change |
|---|---|
| `.gitignore` | Removed the `paper/` and `submission/` rules |
| `LICENSE-PAPER` | Scoped to documentation and figures; no manuscript source |
| `MANIFEST.sha256` | Regenerated for the filtered tree |
| `README.md` | Retitled for the public repository; removed manuscript build instructions |
| `docs/REPRODUCE.md` | Removed the paper build section |
| `docs/MODEL-LIBRARY-DESIGN.md` | Removed a manuscript reference (wording only) |
| `docs/RELEASE-v1.3.0.md` | Removed submission references (wording only) |

Removed paths:

- `v1.2.0`: the 14 files under `paper/` (source, bibliography, figures, built
  PDFs, including the anonymous review copy).
- `v1.3.0`: the same 14 `paper/` files plus `tools/verify_submission_pdf.py`,
  which only checked the submission PDF.

Public-only paths at `v1.3.0`: `CHANGELOG.md`, `CONTRIBUTING.md`,
`SECURITY.md`.

No commit in this repository contains a path under `paper/` or `submission/`.
That can be checked with:

```bash
git log --all --oneline -- 'paper/**' 'submission/**'
```

which returns nothing.

## Tagged metadata is frozen

Tags are not moved. In particular, the `v1.2.0` tag still contains the
`CITATION.cff` that was written before the repository was renamed: its title and
`repository-code` field name `esp32s3-int8-transformer`. The current metadata on
the default branch uses `esp32s3-inference` and is the one to cite. The same
applies to the affiliation wording in both tags.

This is a deliberate trade: the evaluated tree stays byte-identical, and the
metadata correction lives on the default branch instead of rewriting a tag that
a manuscript and a manifest refer to.

## Manifests

`MANIFEST.sha256` covers the filtered tree and excludes itself. A fresh clone
verifies cleanly:

| Tag | Files in manifest | Result |
|---|---:|---|
| `v1.2.0` | 127 | 127/127 OK |
| `v1.3.0` | 166 | 166/166 OK |
| `v1.3.1` | 168 | 168/168 OK |

```bash
git clone https://github.com/dhruvkotian/esp32s3-inference
cd esp32s3-inference
git checkout v1.2.0
sha256sum -c MANIFEST.sha256
```

## Zenodo archive

`v1.3.1` is the archived release:

| Version DOI | Concept DOI | Record |
|---|---|---|
| `10.5281/zenodo.24555021` | `10.5281/zenodo.24555020` | <https://zenodo.org/records/24555021> |

Zenodo archives releases created after the repository integration was enabled,
so `v1.2.0` and `v1.3.0` are not archived. The ZIP for `v1.3.1` is built from the
tag, which predates the `doi` field added to `CITATION.cff` on `main` afterwards.
The tag is not moved; the DOI on `main`, in the release notes and in
`CITATION.cff` is the citation target.
