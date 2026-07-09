"""Module for base_models -> diffusion_model -> video -> wan -> registry.py functionality."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import package_module_root as package_root


@dataclass(frozen=True)
class WanVariant:
    """Static metadata for one Wan implementation variant.

    ``package`` is the canonical package root used for resolving files owned by
    the variant. Migrated forks share the base Wan package and expose concrete
    implementation modules through ``modules``.
    """

    variant_id: str
    display_name: str
    package: str
    base_family: str
    aliases: tuple[str, ...] = ()
    notes: str = ""
    modules: Mapping[str, str] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        """Root.

        Returns:
            The return value.
        """
        return package_root(self.package)

    def module(self, component: str) -> str:
        """Module.

        Args:
            component: The component.

        Returns:
            The return value.
        """
        try:
            return self.modules[component]
        except KeyError as exc:
            available = ", ".join(sorted(self.modules)) or "<none>"
            raise KeyError(
                f"Wan variant {self.variant_id!r} has no component {component!r}. "
                f"Available components: {available}"
            ) from exc


_WAN_VARIANT_LIST: tuple[WanVariant, ...] = (
    WanVariant(
        variant_id="wan2.1",
        display_name="Wan 2.1",
        package="worldfoundry.base_models.diffusion_model.video.wan.wan_2p1",
        base_family="wan2.1",
        aliases=("wan21", "wan2p1", "wan-2.1", "wan_2p1", "2.1"),
        modules={
            "attention": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.attention",
            "clip": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.clip",
            "model": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.model",
            "t5": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.t5",
            "vae": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.vae",
        },
    ),
    WanVariant(
        variant_id="wan2.2",
        display_name="Wan 2.2",
        package="worldfoundry.base_models.diffusion_model.video.wan.wan_2p2",
        base_family="wan2.2",
        aliases=("wan22", "wan2p2", "wan-2.2", "wan_2p2", "2.2"),
        modules={
            "model": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.model",
            "vae2.1": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.vae2_1",
            "vae2.2": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.vae2_2",
        },
    ),
    WanVariant(
        variant_id="matrix-game-2",
        display_name="Matrix-Game-2 Wan",
        package="worldfoundry.base_models.diffusion_model.video.wan",
        base_family="wan2.1",
        aliases=("matrix-game2", "matrix_game_2", "wan_matrix_game_2"),
        notes="Action-control Wan2.1 fork used by Matrix-Game-2.",
        modules={
            "action": "worldfoundry.base_models.diffusion_model.video.wan.components.action_conditioning_wan2p1",
            "attention": "worldfoundry.base_models.diffusion_model.video.wan.components.action_attention_wan2p1",
            "causal_model": "worldfoundry.base_models.diffusion_model.video.wan.models.causal_action_wan2p1",
            "clip": "worldfoundry.base_models.diffusion_model.video.wan.media_encoders.action_clip",
            "image2video": "worldfoundry.base_models.diffusion_model.video.wan.pipelines.action_image2video",
            "model": "worldfoundry.base_models.diffusion_model.video.wan.models.action_wan2p1",
            "prompt_extend": "worldfoundry.base_models.diffusion_model.video.wan.pipelines.action_prompt_extend",
            "text2video": "worldfoundry.base_models.diffusion_model.video.wan.pipelines.action_text2video",
            "vae": "worldfoundry.base_models.diffusion_model.video.wan.vae.action_wan2p1",
        },
    ),
    WanVariant(
        variant_id="matrix-game-3",
        display_name="Matrix-Game-3 Wan",
        package="worldfoundry.base_models.diffusion_model.video.wan",
        base_family="wan2.2",
        aliases=("matrix-game3", "matrix_game_3", "wan_matrix_game_3", "Skywork/Matrix-Game-3.0"),
        notes="Action-control Wan2.2 fork used by Matrix-Game-3.",
        modules={
            "action": "worldfoundry.base_models.diffusion_model.video.wan.components.action_conditioning_wan2p2",
            "attention": "worldfoundry.base_models.diffusion_model.video.wan.components.action_attention_wan2p2",
            "configs": "worldfoundry.base_models.diffusion_model.video.wan.configs.action_wan2p2",
            "model": "worldfoundry.base_models.diffusion_model.video.wan.models.action_wan2p2",
            "t5": "worldfoundry.base_models.diffusion_model.video.wan.components.full_context_t5",
            "triton_kernels": "worldfoundry.base_models.diffusion_model.video.wan.components.int8_triton_kernels",
            "vae": "worldfoundry.base_models.diffusion_model.video.wan.vae.light_wan2p2",
        },
    ),
    WanVariant(
        variant_id="inspatio-world",
        display_name="InSpatio-World Wan",
        package="worldfoundry.base_models.diffusion_model.video.wan",
        base_family="wan2.1",
        aliases=("inspatio", "inspatio_world", "wan_inspatio_world"),
        notes="Causal camera-control Wan2.1 fork used by InSpatio-World.",
        modules={
            "attention": "worldfoundry.base_models.diffusion_model.video.wan.components.camera_attention",
            "causal_model": "worldfoundry.base_models.diffusion_model.video.wan.models.causal_camera_wan2p1",
            "model": "worldfoundry.base_models.diffusion_model.video.wan.models.camera_wan2p1",
            "sage": "worldfoundry.base_models.diffusion_model.video.wan.components.sage_attention",
            "vae": "worldfoundry.base_models.diffusion_model.video.wan.vae.camera_wan2p1",
        },
    ),
    WanVariant(
        variant_id="fantasy-world",
        display_name="FantasyWorld Wan",
        package="worldfoundry.base_models.diffusion_model.video.wan",
        base_family="wan2.1/wan2.2",
        aliases=("fantasy", "fantasy_world", "wan_fantasy_world"),
        notes="FantasyWorld Wan modules used by the paired Wan2.1 and Wan2.2 runners.",
        modules={
            "attention": "worldfoundry.base_models.diffusion_model.video.wan.components.geometry_attention",
            "model": "worldfoundry.base_models.diffusion_model.video.wan.models.geometry_wan",
            "vae": "worldfoundry.base_models.diffusion_model.video.wan.vae.geometry_bridge",
        },
    ),
    WanVariant(
        variant_id="dreamzero",
        display_name="DreamZero Wan",
        package="worldfoundry.base_models.diffusion_model.video.wan.wan_dreamzero",
        base_family="wan2.1",
        aliases=("wan_dreamzero",),
        notes="DreamZero action-generation Wan modules.",
        modules={
            "model": "worldfoundry.base_models.diffusion_model.video.wan.wan_dreamzero.modules.wan_video_dit",
            "vae": "worldfoundry.base_models.diffusion_model.video.wan.wan_dreamzero.modules.wan_video_vae",
        },
    ),
    WanVariant(
        variant_id="sana",
        display_name="Sana Wan",
        package="worldfoundry.base_models.diffusion_model.video.wan",
        base_family="wan",
        aliases=("sana-wan", "wan_sana"),
        notes="Wan components embedded by Sana video/image runtimes.",
        modules={
            "attention": "worldfoundry.base_models.diffusion_model.video.wan.components.linear_attention",
            "clip": "worldfoundry.base_models.diffusion_model.video.wan.media_encoders.linear_clip",
            "model": "worldfoundry.base_models.diffusion_model.video.wan.models.linear_wan",
            "model_wrapper": "worldfoundry.base_models.diffusion_model.video.wan.models.linear_wan_adapter",
            "t5": "worldfoundry.base_models.diffusion_model.video.wan.components.core_loader_t5",
            "vae": "worldfoundry.base_models.diffusion_model.video.wan.vae.linear_wan",
        },
    ),
)


WAN_VARIANTS: Mapping[str, WanVariant] = {
    variant.variant_id: variant for variant in _WAN_VARIANT_LIST
}


def _normalize_variant_key(value: str) -> str:
    """Helper function to normalize variant key.

    Args:
        value: The value.

    Returns:
        The return value.
    """
    return value.strip().lower().replace("_", "-")


def _build_aliases() -> dict[str, WanVariant]:
    """Helper function to build aliases.

    Returns:
        The return value.
    """
    aliases: dict[str, WanVariant] = {}
    for variant in _WAN_VARIANT_LIST:
        for alias in (variant.variant_id, *variant.aliases):
            key = _normalize_variant_key(alias)
            existing = aliases.get(key)
            if existing is not None and existing != variant:
                raise RuntimeError(
                    f"Wan variant alias {alias!r} is registered for both "
                    f"{existing.variant_id!r} and {variant.variant_id!r}."
                )
            aliases[key] = variant
    return aliases


_WAN_VARIANT_ALIASES = _build_aliases()


def available_wan_variants() -> tuple[str, ...]:
    """Available wan variants.

    Returns:
        The return value.
    """
    return tuple(WAN_VARIANTS)


def get_wan_variant(variant: str | WanVariant) -> WanVariant:
    """Get wan variant.

    Args:
        variant: The variant.

    Returns:
        The return value.
    """
    if isinstance(variant, WanVariant):
        return variant
    key = _normalize_variant_key(str(variant))
    try:
        return _WAN_VARIANT_ALIASES[key]
    except KeyError as exc:
        available = ", ".join(available_wan_variants())
        raise ValueError(f"Unknown Wan variant {variant!r}. Available variants: {available}") from exc


def wan_variant_package(variant: str | WanVariant) -> str:
    """Wan variant package.

    Args:
        variant: The variant.

    Returns:
        The return value.
    """
    return get_wan_variant(variant).package


def wan_variant_root(variant: str | WanVariant) -> Path:
    """Wan variant root.

    Args:
        variant: The variant.

    Returns:
        The return value.
    """
    return get_wan_variant(variant).root


def wan_variant_module(variant: str | WanVariant, component: str) -> str:
    """Wan variant module.

    Args:
        variant: The variant.
        component: The component.

    Returns:
        The return value.
    """
    return get_wan_variant(variant).module(component)


def import_wan_variant_symbol(variant: str | WanVariant, component: str, symbol: str) -> Any:
    """Import wan variant symbol.

    Args:
        variant: The variant.
        component: The component.
        symbol: The symbol.

    Returns:
        The return value.
    """
    module = import_module(wan_variant_module(variant, component))
    return getattr(module, symbol)


__all__ = [
    "WAN_VARIANTS",
    "WanVariant",
    "available_wan_variants",
    "get_wan_variant",
    "import_wan_variant_symbol",
    "wan_variant_package",
    "wan_variant_module",
    "wan_variant_root",
]
