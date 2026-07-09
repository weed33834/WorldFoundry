"""OpenEnvision brand rendering for the WorldFoundry TUI.

Renders the official logo PNG as high-resolution Braille terminal art.
Provides deep integration for dark-mode terminals by automatically inverting
the black/dark-grey shades to white/light-grey while preserving the red accent.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    Image = None
    np = None

# ── Brand constants ───────────────────────────────────────────────
BRAND_NAME = "OpenEnvision"
PRODUCT_NAME = "WorldFoundry"
PRODUCT_TAGLINE = "Model · Benchmark · Studio"
LOGO_BACKGROUND = "#ffffff"

# ── Braille dot encoding ──────────────────────────────────────────
# Braille dot positions as (row, col, bit_index) triples for 2×4 dot rendering.
_BRAILLE_DOTS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0), (1, 0, 1), (2, 0, 2), (0, 1, 3),
    (1, 1, 4), (2, 1, 5), (3, 0, 6), (3, 1, 7),
)


# ── Asset discovery ────────────────────────────────────────────────

def logo_asset_path() -> Path:
    """Locate the OpenEnvision logo PNG asset from the package resources directory.

    Falls back to a sibling ``assets/`` directory if package resources are unavailable.
    """
    try:
        asset = resources.files("worldfoundry.cli.assets").joinpath("openenvision_logo.png")
        with resources.as_file(asset) as path:
            return Path(path)
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        return Path(__file__).with_name("assets") / "openenvision_logo.png"


# ── Pixel helpers ──────────────────────────────────────────────────

def _rgb_to_hex(red: int, green: int, blue: int) -> str:
    """Convert RGB channel values to a ``#rrggbb`` hex colour string."""
    return f"#{red:02x}{green:02x}{blue:02x}"


def _prepare_logo_image(image: Image.Image) -> Image.Image:
    """Crop the logo image to its non-white bounding box with padding.

    Converts to RGBA, finds non-white pixels, and returns a padded crop
    region for efficient Braille rendering.
    """
    rgba = image.convert("RGBA")
    pixels = np.array(rgba)
    mask = (pixels[:, :, 0] < 250) | (pixels[:, :, 1] < 250) | (pixels[:, :, 2] < 250)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return rgba

    padding = 6
    left = max(int(xs.min()) - padding, 0)
    top = max(int(ys.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, rgba.width)
    bottom = min(int(ys.max()) + padding + 1, rgba.height)
    return rgba.crop((left, top, right, bottom))


# ── Braille rendering ──────────────────────────────────────────────

def render_logo_braille(image: Image.Image, *, width_chars: int = 56) -> str:
    """Render the logo using Braille characters for highest terminal resolution.

    Each Braille character covers a 2×4 pixel block, yielding much finer
    detail than half-block rendering. Dark-mode terminals benefit from
    automatic colour inversion — black/dark-grey shades are flipped to
    white/light-grey while the red accent is preserved.

    Args:
        image: Source logo image (typically RGBA PNG).
        width_chars: Target width in terminal character columns.

    Returns:
        Rich-markup string with per-cell ``[color]`` annotations.

    Raises:
        RuntimeError: When Pillow is not installed.
    """
    if Image is None or np is None:
        raise RuntimeError("Pillow is required to render the OpenEnvision logo.")

    # ── Prepare and resize ──
    prepared = _prepare_logo_image(image)
    source_width, source_height = prepared.size
    
    # 1 Braille char = 2 pixels wide, 4 pixels high
    pixel_width = width_chars * 2
    pixel_height = int(source_height * (pixel_width / source_width))
    remainder = pixel_height % 4
    if remainder:
        pixel_height += 4 - remainder

    resized = prepared.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
    pixels = np.array(resized.convert("RGBA"))
    
    char_height = pixel_height // 4
        
    lines: list[str] = []

    # ── Iterate character cells ──
    for char_row in range(char_height):
        row_y = char_row * 4
        parts: list[str] = []
        for char_col in range(width_chars):
            col_x = char_col * 2
            # NOTE: Guard against rounding errors near edges
            if row_y >= pixels.shape[0] or col_x >= pixels.shape[1]:
                parts.append(" ")
                continue
                
            block = pixels[row_y : min(row_y + 4, pixels.shape[0]), 
                           col_x : min(col_x + 2, pixels.shape[1])]

            # ── Compute Braille bits and colour ──
            bits = 0
            colors: list[tuple[int, int, int]] = []
            for py, px, bit in _BRAILLE_DOTS:
                if py >= block.shape[0] or px >= block.shape[1]:
                    continue
                r, g, b, a = block[py, px]
                # Skip transparent or near-white pixels
                if a < 20 or (r > 240 and g > 240 and b > 240):
                    continue
                
                bits |= 1 << bit
                # NOTE: Dark-mode inversion — keep the red accent, invert everything else
                if r > 150 and g < 100 and b < 100:
                    colors.append((r, g, b))
                else:
                    colors.append((255 - r, 255 - g, 255 - b))

            if bits == 0:
                parts.append(" ")
                continue

            # ── Average colour and emit Rich-markup cell ──
            avg_r = sum(c[0] for c in colors) // len(colors)
            avg_g = sum(c[1] for c in colors) // len(colors)
            avg_b = sum(c[2] for c in colors) // len(colors)
            
            hex_color = _rgb_to_hex(avg_r, avg_g, avg_b)
            char = chr(0x2800 + bits)
            parts.append(f"[{hex_color}]{char}[/]")

        lines.append("".join(parts))
    return "\n".join(lines)


# ── Brand rendering ────────────────────────────────────────────────

@lru_cache(maxsize=4)
def render_brand_logo(*, width_chars: int = 56) -> str:
    """Render the OpenEnvision logo as Braille-art with dark-mode inversion.

    Caches up to 4 width variants. Reads the PNG asset via :func:`logo_asset_path`.

    Raises:
        FileNotFoundError: When the logo asset PNG is not found.
        RuntimeError: When Pillow is not installed.
    """
    path = logo_asset_path()
    if not path.is_file():
        raise FileNotFoundError(f"OpenEnvision logo asset not found: {path}")
    with Image.open(path) as image:
        return render_logo_braille(image, width_chars=width_chars)


def render_fallback_header(*, width_chars: int = 56) -> str:
    """Return a plain-text header for the ``--fallback`` mode (no Textual dependency).

    Falls back to simple text lines if Braille rendering fails.
    """
    try:
        logo = render_brand_logo(width_chars=width_chars)
        return f"\n{logo}\n\n        {PRODUCT_NAME} · {PRODUCT_TAGLINE}\n"
    except Exception:
        return "\n".join([BRAND_NAME, PRODUCT_NAME, PRODUCT_TAGLINE, ""])
