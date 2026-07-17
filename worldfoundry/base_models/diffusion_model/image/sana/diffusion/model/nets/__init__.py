"""Sana network registrations.

The general package keeps the historical eager registration behaviour.  A
dedicated inference entry point may select the ``v2v`` profile to avoid
importing unrelated image, ControlNet, camera-control, FLA, and Triton modules
on every subprocess startup.
"""

import os

_PROFILE = os.environ.get("WORLDFOUNDRY_SANA_NETS_PROFILE", "").strip().lower()

if _PROFILE == "v2v":
    from . import sana_v2v_attn_blocks
    from .sana_multi_scale_video_v2v import SanaMSVideoV2V, SanaMSVideoV2V_2000M_P1_D20
elif _PROFILE == "wm":
    # SANA-WM only needs the camera-conditioned 1.6B video network. Avoid
    # importing image, ControlNet, U-shaped, and V2V registrations on every
    # interactive process startup.
    from . import sana_gdn_blocks, sana_gdn_blocks_triton, sana_gdn_camctrl_blocks
    from .sana_multi_scale_video_camctrl import (
        SanaMSVideoCamCtrl,
        SanaMSVideoCamCtrl_1600M_P1_D20,
    )
else:
    from . import sana_gdn_blocks, sana_gdn_blocks_triton, sana_gdn_camctrl_blocks, sana_v2v_attn_blocks
    from .sana import (
        Sana,
        SanaBlock,
        get_1d_sincos_pos_embed_from_grid,
        get_2d_sincos_pos_embed,
        get_2d_sincos_pos_embed_from_grid,
    )
    from .sana_multi_scale import (
        SanaMS,
        SanaMS_600M_P1_D28,
        SanaMS_600M_P2_D28,
        SanaMS_600M_P4_D28,
        SanaMS_1600M_P1_D20,
        SanaMS_1600M_P2_D20,
        SanaMSBlock,
    )
    from .sana_multi_scale_controlnet import SanaMSControlNet_600M_P1_D28
    from .sana_multi_scale_video import (
        SanaMSVideo,
        SanaMSVideo_600M_P1_D28,
        SanaMSVideo_600M_P2_D28,
        SanaMSVideo_2000M_P1_D20,
        SanaMSVideo_2000M_P2_D20,
    )
    from .sana_multi_scale_video_v2v import SanaMSVideoV2V, SanaMSVideoV2V_2000M_P1_D20
    from .sana_multi_scale_video_camctrl import (
        SanaMSVideoCamCtrl,
        SanaMSVideoCamCtrl_1600M_P1_D20,
    )
