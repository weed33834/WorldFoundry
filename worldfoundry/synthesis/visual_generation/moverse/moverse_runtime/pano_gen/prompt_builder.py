"""Prompt building utilities for panorama and video generation.

No caption model is used — all prompts come from user input.
"""

# Prepended to every FluxFill prompt (LoRA trigger words)
TRIGGER_WORDS = "Equirectangular 360 panorama."

# Appended to every FluxFill prompt (quality boosters)
FLUX_SUFFIX = ", high-quality, high resolution, sharp and detailed, clear, 8k, HDRI, Ultra HD"


def build_pano_prompt(prompt: str, back_prompt: str = "") -> str:
    """Build the full FluxFill prompt from user inputs.

    Assembly order:
        TRIGGER_WORDS + " " + prompt + [", " + back_prompt] + FLUX_SUFFIX

    Args:
        prompt: User's main prompt for the panorama (required).
        back_prompt: Optional description of the unseen 270° content.

    Returns:
        Full prompt string ready for FluxFill.
    """
    clip_prompt = TRIGGER_WORDS + " " + prompt
    back_suffix = (", " + back_prompt) if back_prompt.strip() else ""
    return clip_prompt + back_suffix + FLUX_SUFFIX


def derive_video_prompt(pano_prompt: str) -> str:
    """Derive a video generation prompt from the panorama prompt.

    The Wan2.1 T2V model receives: "A video of " + <user's pano prompt>

    Args:
        pano_prompt: The user's main panorama prompt (without trigger words / suffix).

    Returns:
        Video prompt string for Wan2.1 T2V.
    """
    return "A video of " + pano_prompt
