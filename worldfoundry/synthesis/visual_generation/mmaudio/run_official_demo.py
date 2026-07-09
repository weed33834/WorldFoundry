from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any


def _patch_torchaudio_save() -> None:
    import soundfile as sf
    import torchaudio

    def _save_with_soundfile(
        uri: str | Path,
        src: Any,
        sample_rate: int,
        channels_first: bool = True,
        format: str | None = None,
        **_: Any,
    ) -> None:
        tensor = src.detach().cpu() if hasattr(src, "detach") else src
        if hasattr(tensor, "numpy"):
            data = tensor.numpy()
        else:
            data = tensor
        if channels_first and getattr(data, "ndim", 0) == 2:
            data = data.T
        sf.write(str(uri), data, int(sample_rate), format=format)

    torchaudio.save = _save_with_soundfile  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worldfoundry-demo-script", required=True)
    args, remaining = parser.parse_known_args(argv)

    demo_script = Path(args.worldfoundry_demo_script).expanduser().resolve()
    if not demo_script.is_file():
        raise FileNotFoundError(f"MMAudio demo script not found: {demo_script}")
    sys.path.insert(0, str(demo_script.parent))

    _patch_torchaudio_save()
    sys.argv = [str(demo_script), *remaining]
    runpy.run_path(str(demo_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
