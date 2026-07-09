from __future__ import annotations

from html import escape
from typing import Iterable


def hero_html(*,
    stats: dict[str, int],
    filtered_total: int,
    selected_title: str,
    category: str,
    search: str,
) -> str:
    scope_bits = []
    if category and category != "All":
        scope_bits.append(category)
    if search:
        scope_bits.append(f"search: {search}")
    scope_text = " · ".join(scope_bits) if scope_bits else "all pipelines"
    return f"""
<section class="wa-hero">
  <div class="wa-hero-copy">
    <div class="wa-hero-eyebrow">WorldFoundry Studio</div>
    <h1>{escape(selected_title)}</h1>
    <p>CLI-first interactive stage for world-model demos.</p>
  </div>
  <div class="wa-hero-context">
    <div class="wa-hero-focus">
      <div class="wa-hero-focus-label">Scope</div>
      <strong>{filtered_total} model(s)</strong>
      <span>{escape(scope_text)}</span>
    </div>
    <div class="wa-metric-grid">
      <div class="wa-metric"><strong>{stats.get("total", 0)}</strong><span>pipelines discovered</span></div>
      <div class="wa-metric"><strong>{stats.get("video", 0)}</strong><span>video generation</span></div>
      <div class="wa-metric"><strong>{stats.get("scene", 0)}</strong><span>3D / geometry</span></div>
      <div class="wa-metric"><strong>{stats.get("stream", 0)}</strong><span>stream-capable</span></div>
    </div>
  </div>
</section>
"""


def profile_html(title: str, summary: str, pills: list[str], notes: str = "") -> str:
    pill_html = "".join(f'<span class="wa-pill">{escape(pill)}</span>' for pill in pills if pill)
    notes_html = f'<div class="wa-runnote">{escape(notes)}</div>' if notes else ""
    return f"""
<section class="wa-profile">
  <div class="wa-profile-eyebrow">Selected Model</div>
  <h3>{escape(title)}</h3>
  <p>{escape(summary)}</p>
  <div class="wa-pillrow">{pill_html}</div>
  {notes_html}
</section>
"""


def summary_html(
    title: str,
    subtitle: str = "",
    pills: Iterable[str] = (),
    lines: Iterable[str] = (),
    variant: str = "default",
) -> str:
    pill_html = "".join(
        f'<span class="wa-summary-pill">{escape(pill)}</span>' for pill in pills if pill
    )
    lines_html = "".join(f"<div>{escape(line)}</div>" for line in lines if line)
    variant_class = " is-danger" if variant == "danger" else ""
    subtitle_html = f'<div class="wa-summary-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    lines_block = f'<div class="wa-summary-lines">{lines_html}</div>' if lines_html else ""
    pills_block = f'<div class="wa-summary-pills">{pill_html}</div>' if pill_html else ""
    return f"""
<section class="wa-summary-card{variant_class}">
  <h4>{escape(title)}</h4>
  {subtitle_html}
  {pills_block}
  {lines_block}
</section>
"""

__all__ = ["hero_html", "profile_html", "summary_html"]
