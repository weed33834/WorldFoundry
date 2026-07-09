"""CLI entry-point for ``python -m worldfoundry.cli``.

Delegates to :func:`worldfoundry.cli.main.main` and exits with its return code.
"""

from .main import main

# ── CLI bootstrap ──────────────────────────────────────────────────
if __name__ == "__main__":
    raise SystemExit(main())
