#!/usr/bin/env python3
"""Inline metric reference rows into metrics-usage partials and drop duplicate sections."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "fumadocs"
METRICS_EN = DOCS / "content/docs/evaluation/metrics.mdx"
METRICS_ZH = DOCS / "content/docs/evaluation/metrics.zh.mdx"
USAGE_EN = DOCS / "mdx/partials/metrics-usage.mdx"
USAGE_ZH = DOCS / "mdx/partials/metrics-usage.zh.mdx"

REFERENCE_HEADING_EN = "## Metric reference"
REFERENCE_HEADING_ZH = "## 指标参考"


def _parse_reference_tables(text: str) -> dict[str, tuple[str, str, str]]:
    refs: dict[str, tuple[str, str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 4 or cells[0].lower() in {"metric", "指标"}:
            continue
        metric_id = cells[0].strip("`").strip()
        refs[metric_id] = (cells[1], cells[2], cells[3])
    return refs


def _strip_reference_section(text: str, heading: str) -> str:
    if heading not in text:
        return text
    before, rest = text.split(heading, 1)
    next_heading = re.search(r"\n## ", rest)
    if next_heading:
        after = rest[next_heading.start() :]
    else:
        after = ""
    return before.rstrip() + "\n\n" + after.lstrip("\n")


def _format_reference(principle: str, paper: str, repo: str, *, locale: str) -> str:
    label = "Reference" if locale == "en" else "参考"
    parts = [f"**{label}:** {principle}"]
    if paper and paper != "—":
        parts.append(f"Paper: {paper}" if locale == "en" else f"论文: {paper}")
    if repo and repo != "—":
        parts.append(f"Repo: {repo}" if locale == "en" else f"仓库: {repo}")
    return " ".join(parts)


def _metric_ids_from_heading(heading: str) -> list[str]:
    text = heading.strip()
    text = re.sub(r"^###\s+", "", text)
    text = re.sub(r"\s*\([↑↓]\)\s*$", "", text)
    text = text.strip("`").strip()
    ids: list[str] = []
    for part in text.split("/"):
        token = part.strip().strip("`")
        if not token:
            continue
        base = token.split(":", 1)[0].strip()
        if base == "numeric":
            ids.extend(["numeric", "numeric_value"])
        elif base == "has_artifact":
            ids.append("has_artifact")
        else:
            ids.append(base)
    return list(dict.fromkeys(ids))


def _reference_block(metric_ids: list[str], refs: dict[str, tuple[str, str, str]], *, locale: str) -> str:
    lines: list[str] = []
    for metric_id in metric_ids:
        row = refs.get(metric_id)
        if not row:
            continue
        principle, paper, repo = row
        if len(metric_ids) == 1:
            lines.append(_format_reference(principle, paper, repo, locale=locale))
        else:
            prefix = "Reference" if locale == "en" else "参考"
            lines.append(f"- **`{metric_id}`:** {_format_reference(principle, paper, repo, locale=locale).replace(f'**{prefix}:** ', '', 1)}")
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    header = "**Reference:**" if locale == "en" else "**参考：**"
    return header + "\n" + "\n".join(lines)


def _inline_references(usage_text: str, refs: dict[str, tuple[str, str, str]], *, locale: str) -> str:
    when_markers = ("**When**:", "**用法**:")
    ref_prefixes = ("**Reference:**", "**参考：**", "**参考:**")
    out: list[str] = []
    lines = usage_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### ") and not line.startswith("### Run multiple"):
            metric_ids = _metric_ids_from_heading(line)
            out.append(line)
            i += 1
            inserted = False
            while i < len(lines) and not lines[i].startswith("### ") and not lines[i].startswith("## "):
                current = lines[i]
                if any(current.startswith(prefix) for prefix in ref_prefixes):
                    i += 1
                    while i < len(lines) and (
                        lines[i].startswith("- **`") or lines[i].strip() == ""
                    ):
                        i += 1
                    continue
                if not inserted and any(current.startswith(marker) for marker in when_markers):
                    out.append(current)
                    block = _reference_block(metric_ids, refs, locale=locale)
                    if block:
                        out.append("")
                        out.extend(block.splitlines())
                    inserted = True
                    i += 1
                    continue
                out.append(current)
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out).rstrip() + "\n"


def _remove_existing_reference_blocks(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("**Reference:**") or line.startswith("**参考：**") or line.startswith("**参考:**"):
            i += 1
            while i < len(lines) and (lines[i].startswith("- **`") or lines[i].strip() == ""):
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def main() -> None:
    metrics_en = METRICS_EN.read_text(encoding="utf-8")
    metrics_zh = METRICS_ZH.read_text(encoding="utf-8")
    refs_en = _parse_reference_tables(metrics_en)
    refs_zh = _parse_reference_tables(metrics_zh)

    for usage_path, refs, locale in ((USAGE_EN, refs_en, "en"), (USAGE_ZH, refs_zh, "zh")):
        cleaned = _remove_existing_reference_blocks(usage_path.read_text(encoding="utf-8"))
        usage_path.write_text(_inline_references(cleaned, refs, locale=locale), encoding="utf-8")

    METRICS_EN.write_text(_strip_reference_section(metrics_en, REFERENCE_HEADING_EN), encoding="utf-8")
    METRICS_ZH.write_text(_strip_reference_section(metrics_zh, REFERENCE_HEADING_ZH), encoding="utf-8")

    for path, old, new in (
        (METRICS_EN, "Documented in [Metric reference](#metric-reference) above", "Listed inline under each metric in [How to use each metric](#how-to-use-each-metric)"),
        (METRICS_ZH, "见上文 [指标参考](#指标参考)", "见 [每个指标怎么用](#每个指标怎么用) 各 metric 小节内联说明"),
    ):
        text = path.read_text(encoding="utf-8")
        if old in text:
            path.write_text(text.replace(old, new), encoding="utf-8")

    print(f"Inlined references into {USAGE_EN.name} and {USAGE_ZH.name}")


if __name__ == "__main__":
    main()
