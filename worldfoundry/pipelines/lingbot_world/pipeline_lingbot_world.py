"""Lingbot World visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import numpy as np
from PIL import Image
from typing import Optional, Any, Dict, Union

from ...operators.lingbot_world_operator import LingBotOperator
from ...synthesis.visual_generation.lingbot_world.lingbot_world_synthesis import LingBotSynthesis
from ...synthesis.visual_generation.memory.stream import VisualFrameMemory
from ...synthesis.visual_generation.lingbot_world.lingbot_world_runtime.utils.wasd_ijkl_to_c2ws import (
    generate_and_save_trajectory,
)


WMFACTORY_BASE_INTRINSICS = np.asarray(
    [415.5298, 415.6922, 415.77786, 239.77779],
    dtype=np.float32,
)
WMFACTORY_INTERACTION_KEYS: Dict[str, tuple[str, ...]] = {
    "forward": ("w",),
    "backward": ("s",),
    "left": ("a",),
    "right": ("d",),
    "forward_left": ("w", "a"),
    "forward_right": ("w", "d"),
    "backward_left": ("s", "a"),
    "backward_right": ("s", "d"),
    "camera_up": ("i",),
    "camera_down": ("k",),
    "camera_l": ("j",),
    "camera_r": ("l",),
    "camera_ul": ("i", "j"),
    "camera_ur": ("i", "l"),
    "camera_dl": ("k", "j"),
    "camera_dr": ("k", "l"),
}


def _wmfactory_trajectory_from_interactions(
    interactions: Optional[list[str]],
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Wmfactory trajectory from interactions helper function."""
    keys: list[str] = []
    unknown: list[str] = []
    for raw_token in interactions or []:
        token = str(raw_token).strip().lower()
        if not token:
            continue
        mapped = WMFACTORY_INTERACTION_KEYS.get(token)
        if mapped is None:
            unknown.append(token)
            continue
        for key in mapped:
            if key not in keys:
                keys.append(key)

    if unknown:
        unsupported = ", ".join(sorted(set(unknown)))
        raise ValueError(f"Unsupported LingBot WMFactory interaction token(s): {unsupported}")

    repeat_frames = max(int(num_frames) - 1, 1)
    frame_keys = [keys[:] for _ in range(repeat_frames)]
    c2ws = np.asarray(generate_and_save_trajectory(frame_keys), dtype=np.float32)
    intrinsics = np.repeat(WMFACTORY_BASE_INTRINSICS[None, :], c2ws.shape[0], axis=0).astype(np.float32)
    return c2ws, intrinsics


class LingBotPipeline(PipelineABC):
    """Pipeline implementation for LingBot visual generation."""
    def __init__(self,
                 operators: Optional[LingBotOperator] = None,
                 synthesis_model: Optional[LingBotSynthesis] = None,
                 memory_module: Optional[Any] = None,
                 device: str = "cuda"
                 ):
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.operators = operators
        self.memory_module = memory_module
        self.device = device

    @classmethod
    def from_pretrained(cls,
                        model_path: Union[str, Dict[str, Any], None] = None,
                        required_components: Optional[Dict[str, Any]] = None,
                        mode: str = "i2v-A14B",
                        device: str = "cuda",
                        **kwargs) -> "LingBotPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        component_options = dict(required_components or {})
        if isinstance(model_path, dict):
            component_options.update(model_path)
            model_path = component_options.pop("model_path", None)
        mode = component_options.pop("mode", mode)
        kwargs = cls._strip_framework_loading_options({**component_options, **kwargs})
        model_path = model_path or "robbyant/lingbot-world-base-cam"
        
        print(f"Loading LingBot World Model from {model_path}...")
        
        synthesis_model = LingBotSynthesis.from_pretrained(
            pretrained_model_path=model_path,
            task=mode,
            device=device,
            **kwargs
        )
        
        operators = LingBotOperator()
        memory_module = VisualFrameMemory(model_id="lingbot-world")

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device
        )
        return pipeline
    
    def process(self,
                images: Any = None,
                prompt: Optional[str] = None,
                interactions: Optional[list[str]] = None,
                action_path: Optional[str] = None,
                resize_H: int = 480,
                resize_W: int = 832,
                num_frames: Optional[int] = 81,
                wmfactory_action_controls: bool = False): 
        
        """Process and normalize input arguments and conditions for inference."""
        if isinstance(images, str):
            images = Image.open(images).convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("Input image must be a PIL image or an image path.")

        output_dict = {
            "pil_image": images,
            "prompt": prompt if prompt is not None else "",
            "action_path": action_path,
            "c2ws": None,
            "Ks": None,
        }

        if action_path is None and interactions:
            if wmfactory_action_controls:
                c2ws, Ks = _wmfactory_trajectory_from_interactions(interactions, num_frames or 81)
            else:
                c2ws, Ks = self.operators.traj_generator.generate(
                    interactions,
                    num_frames,
                    resize_H,
                    resize_W,
                )
            output_dict["c2ws"] = c2ws
            output_dict["Ks"] = Ks

        return output_dict

    def __call__(self,
                  images: Any = None,
                  num_frames: Optional[int] = 81,
                  prompt: Optional[str] = None,
                  interactions: Optional[list[str]] = None,
                  action_path: Optional[str] = None,
                  resize_H: int = 480,
                  resize_W: int = 832,
                  max_area: int = 720 * 1280,
                  vis_ui: bool = False,
                  allow_act2cam: bool = False,
                  action_string: Optional[str] = None,
                  wmfactory_action_controls: bool = False,
                  seed: int = 42, 
                  **kwds):
        
        """Execute the complete pipeline generation flow."""
        processed_inputs = self.process(
            images=images,
            prompt=prompt,
            interactions=interactions,
            action_path=action_path,
            resize_H=resize_H,
            resize_W=resize_W,
            num_frames=num_frames,
            wmfactory_action_controls=wmfactory_action_controls,
        )
        
        output_video = self.synthesis_model.predict(
            prompt=processed_inputs["prompt"],
            pil_image=processed_inputs["pil_image"],
            action_path=processed_inputs["action_path"],
            c2ws=processed_inputs["c2ws"],
            Ks=processed_inputs["Ks"],
            num_output_frames=num_frames,
            max_area=max_area,
            vis_ui=vis_ui,
            allow_act2cam=allow_act2cam,
            action_string=action_string,
            seed=seed,
            **kwds
        )
        
        return output_video
    
    def stream(self,
                prompt: Optional[str] = None,
                interactions: Optional[list[str]] = None,
                images: Any = None,
                action_path: Optional[str] = None,
                num_frames: Optional[int] = 81,
                resize_H: int = 480,
                resize_W: int = 832,
                max_area: int = 720 * 1280,
                vis_ui: bool = False,
                allow_act2cam: bool = False,
                action_string: Optional[str] = None,
                wmfactory_action_controls: bool = False,
                seed: int = 42,
                **kwds) -> np.ndarray:
        
        # 1. Initialize Memory if images provided (First Turn)
        """Stream visual generation outputs chunk by chunk."""
        if images is not None:
            print("--- Stream Started ---")
            self.memory_module.manage(action="reset") # Clear old memory
            self.memory_module.record(images, type="image")
        
        # 2. Retrieve Context (Input for this turn)
        current_img = self.memory_module.select()
        if current_img is None:
            raise ValueError("No image in storage. Provide 'images' first.")

        # 3. Generate Video
        video_output = self.__call__(
            images=current_img,
            num_frames=num_frames,
            prompt=prompt,
            interactions=interactions,
            action_path=action_path,
            resize_H=resize_H,
            resize_W=resize_W,
            max_area=max_area,
            vis_ui=vis_ui,
            allow_act2cam=allow_act2cam,
            action_string=action_string,
            wmfactory_action_controls=wmfactory_action_controls,
            seed=seed,
            **kwds
        ) # Returns numpy array [T, H, W, C]

        # 4. Record Result (Updates context for next turn)
        if video_output is not None:
            self.memory_module.record(video_output, type="video_chunk")

        return video_output
