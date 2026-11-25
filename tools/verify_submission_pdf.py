#!/usr/bin/env python3
"""Fail loudly if a paper PDF is not the corrected TECS build.

This is a pre-upload gate. It checks the title, page count, corrected metric
wording, the eighteen-run statement, hardware provenance, and the font
embedding. Mode-specific checks cover the anonymous, manuscript, preprint, and
acmsmall builds.

Usage:
    python3 tools/verify_submission_pdf.py --mode anonymous FILE.pdf
    python3 tools/verify_submission_pdf.py --mode manuscript FILE.pdf
    python3 tools/verify_submission_pdf.py --mode preprint FILE.pdf
    python3 tools/verify_submission_pdf.py --mode acmsmall FILE.pdf

Exit status is 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

NEW_TITLE = "Bytes, Not FLOPs: INT8 KV Caching and 128-bit PIE Attention on ESP32-S3"
OLD_TITLE = "INT8 KV-Cache and Attention Optimisation"
EXPECTED_PAGES = 19
ESP_DSP_COMMIT = "a53a0756833c045311ea1d79a2badf495cdfde4c"
TOP1_SENTENCE = (
    "Top-1 predictions match the FP32 reference at all nine evaluation depths, "
    "while the combined relative logit RMSE ranges from 1.08% to 4.56%."
)
DOI = "10.1145/nnnnnnn.nnnnnnn"


def tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(f"required tool not found on PATH: {name}")
    return path


def run(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # ACM proof PDFs interleave standalone line numbers with the text.
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def page_count(path: Path) -> int:
    out = run(tool("pdfinfo"), str(path))
    m = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
    if not m:
        raise ValueError(f"no page count in pdfinfo output for {path}")
    return int(m.group(1))


def text_of(path: Path) -> str:
    return normalize(run(tool("pdftotext"), "-layout", str(path), "-"))


def font_report(path: Path) -> tuple[int, int, int]:
    """Return (rows, not_embedded, type3)."""
    out = run(tool("pdffonts"), str(path))
    lines = out.splitlines()
    if len(lines) < 3:
        return 0, 0, 0
    header = lines[0]
    start = header.index("emb")
    end = header.index("sub")
    rows = not_embedded = type3 = 0
    for line in lines[2:]:
        if not line.strip():
            continue
        rows += 1
        if line[start:end].strip() != "yes":
            not_embedded += 1
        if "Type 3" in line:
            type3 += 1
    return rows, not_embedded, type3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="+", type=Path)
    ap.add_argument("--mode", required=True,
                    choices=["anonymous", "manuscript", "preprint", "acmsmall"])
    ap.add_argument("--pages", type=int, default=EXPECTED_PAGES)
    args = ap.parse_args()

    failures = 0
    for pdf in args.pdf:
        print(f"\n== {pdf} ({args.mode}) ==")
        if not pdf.is_file():
            print("  [FAIL] file does not exist")
            failures += 1
            continue

        text = text_of(pdf)
        pages = page_count(pdf)
        fonts, not_embedded, type3 = font_report(pdf)
        checks: list[tuple[str, bool, str]] = []

        checks.append(("page count", pages == args.pages,
                       f"{pages} pages, expected {args.pages}"))
        checks.append(("new title", NEW_TITLE in text, NEW_TITLE))
        checks.append(("old title absent", OLD_TITLE not in text, OLD_TITLE))
        checks.append(("no 18-page proof header", "Page 1 of 18" not in text,
                       "Page 1 of 18 must not appear"))
        checks.append(("no submission cover page",
                       "Date Submitted by the Author" not in text
                       and "Manuscript Type:" not in text,
                       "ACM submission cover metadata must not appear"))
        checks.append(("no twelve-run statement",
                       "twelve-run" not in text.lower(),
                       "old twelve-run wording must not appear"))
        checks.append(("eighteen-run contribution",
                       "an eighteen-run reproducibility check" in text.lower(),
                       "an eighteen-run reproducibility check must appear"))
        checks.append(("primary 166.6 forwards/s",
                       "166.6 decode forwards/s" in text,
                       "166.6 decode forwards/s must appear"))
        checks.append(("secondary 168.25 sampled tokens/s",
                       "168.25 sampled tokens/s" in text,
                       "168.25 sampled tokens/s must appear"))
        checks.append(("hybrid cache total", "266,240 B" in text,
                       "266,240 B must appear"))
        checks.append(("PSRAM cache total", "245,760 B" in text,
                       "245,760 B must appear"))
        checks.append(("ESP-DSP commit", ESP_DSP_COMMIT in text,
                       ESP_DSP_COMMIT))
        checks.append(("exact top-1 sentence", TOP1_SENTENCE in text,
                       TOP1_SENTENCE[:60] + "..."))
        checks.append(("fonts embedded", not_embedded == 0,
                       f"{fonts} rows, {not_embedded} not embedded"))
        checks.append(("no Type 3 fonts", type3 == 0,
                       f"{type3} Type 3 rows"))

        if args.mode == "anonymous":
            checks += [
                ("anonymous author block", "ANONYMOUS AUTHOR" in text,
                 "ANONYMOUS AUTHOR must appear"),
                ("no author name", "Dhruv Kotian" not in text,
                 "Dhruv Kotian must not appear"),
                ("no institution", "RTMSSU" not in text,
                 "RTMSSU must not appear"),
                ("no personal email", "dhruvjkotian" not in text,
                 "dhruvjkotian must not appear"),
                ("no personal artifact URL",
                 "github.com/dhruvkotian" not in text,
                 "github.com/dhruvkotian must not appear"),
            ]
        else:
            checks.append(("identified author", "Dhruv Kotian" in text,
                           "Dhruv Kotian must appear"))

        if args.mode in ("manuscript", "acmsmall"):
            checks += [
                ("ACM reference format", "ACM Reference Format" in text,
                 "ACM Reference Format must appear"),
                ("DOI placeholder", DOI in text, DOI),
            ]
        elif args.mode == "preprint":
            checks += [
                ("no ACM reference format", "ACM Reference Format" not in text,
                 "ACM Reference Format must not appear"),
                ("no DOI placeholder", DOI not in text,
                 f"{DOI} must not appear"),
            ]

        for name, ok, detail in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
            if not ok:
                failures += 1

    print(f"\n{'OVERALL: PASS' if failures == 0 else f'OVERALL: FAIL ({failures} checks)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
