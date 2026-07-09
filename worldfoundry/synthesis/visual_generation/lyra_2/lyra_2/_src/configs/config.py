from typing import Any, List

import attrs

from lyra_2._ext.imaginaire import config
from lyra_2._ext.imaginaire.utils.config_helper import import_all_modules_from_package
from lyra_2._src.configs.defaults.common.checkpoint import register_checkpoint
from lyra_2._src.configs.defaults.common.ckpt_type import register_ckpt_type
from lyra_2._src.configs.defaults.common.ema import register_ema
from lyra_2._src.configs.defaults.common.tokenizer import register_tokenizer
from lyra_2._src.configs.defaults.conditioner import lyra_register_conditioner
from lyra_2._src.configs.defaults.model import lyra_register_model
from lyra_2._src.configs.defaults.net import lyra_register_net


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "fsdp_wan2pt1_lyra2_spatial"},
            {"net": "wan2pt1_14B_i2v_lyra2"},
            {"conditioner": "lyra2_conditioner"},
            {"ema": "power"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"checkpoint": "local"},
            {"ckpt_type": "dummy"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(model=None)

    c.job.project = "lyra_2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    # Register common defaults
    register_ema()
    register_tokenizer()
    register_checkpoint()
    register_ckpt_type()

    # Register lyra_2-specific configs
    lyra_register_model()
    lyra_register_net()
    lyra_register_conditioner()

    # Register experiment configs
    import_all_modules_from_package("lyra_2._src.configs", reload=True)

    return c
