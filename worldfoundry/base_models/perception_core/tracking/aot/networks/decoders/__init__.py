from worldfoundry.base_models.perception_core.tracking.aot.networks.decoders.fpn import FPNSegmentationHead


def build_decoder(name, **kwargs):

    if name == 'fpn':
        return FPNSegmentationHead(**kwargs)
    else:
        raise NotImplementedError
