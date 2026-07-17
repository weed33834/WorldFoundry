"""ATI's minimal official inference closure.

The trajectory patch and I2V entrypoint are vendored from bytedance/ATI at
revision ``1a002caf7bb55cfb016dcc670c357bd803af3a0d``.  The unchanged Wan2.1
modules are imported from WorldFoundry's central in-tree Wan base model.
"""

from .image2video import WanATI
from .motion import get_tracks_inference, process_tracks, unzip_to_array

__all__ = ["WanATI", "get_tracks_inference", "process_tracks", "unzip_to_array"]
