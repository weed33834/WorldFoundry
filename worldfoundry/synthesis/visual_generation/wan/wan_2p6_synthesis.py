from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List

import requests


RUNTIME_STATUS = {
    "runtime_mode": "api",
    "backend_stage": "external_service",
    "in_tree_backend": False,
    "external_service": True,
}


class Wan2p6Synthesis(object):
    """
    Wan2.6 API 合成类。

    支持文本生成视频、图像生成视频和参考素材生成视频。
    """
    RUNTIME_STATUS = RUNTIME_STATUS
    IN_TREE_BACKEND = False
    BACKEND_STAGE = "external_service"
    EXTERNAL_SERVICE = True

    def __init__(
        self,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
        logger=None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.logger = logger

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ):
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
        )

    def _headers(self, async_request: bool = True) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if async_request:
            headers["X-DashScope-Async"] = "enable"
        return headers

    def _create_url(self) -> str:
        task_path = "/services/aigc/video-generation/video-synthesis"
        if self.endpoint.endswith(task_path):
            return self.endpoint
        return f"{self.endpoint}{task_path}"

    def _task_status_url(self, task_id: str) -> str:
        task_path = "/services/aigc/video-generation/video-synthesis"
        base_endpoint = self.endpoint
        if base_endpoint.endswith(task_path):
            base_endpoint = base_endpoint[:-len(task_path)]
        return f"{base_endpoint}/tasks/{task_id}"

    def _post_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            self._create_url(),
            headers=self._headers(async_request=True),
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def generate_t2av(
        self,
        input_prompt: str,
        model: str = "wan2.6-t2v",
        size: str = "1280*720",
        duration: int = 5,
        negative_prompt: str = "",
        audio_url: Optional[str] = None,
        prompt_extend: bool = True,
        shot_type: Optional[str] = None,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "input": {
                "prompt": input_prompt,
            },
            "parameters": {
                "size": size,
                "duration": duration,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
            },
        }
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        if audio_url:
            payload["input"]["audio_url"] = audio_url
        if shot_type is not None:
            payload["parameters"]["shot_type"] = shot_type
        if seed is not None:
            payload["parameters"]["seed"] = seed
        payload["parameters"].update(kwargs)
        return self._post_task(payload)

    def generate_i2av(
        self,
        encoded_image: str,
        input_prompt: str,
        model: str = "wan2.6-i2v",
        resolution: str = "720P",
        duration: int = 5,
        negative_prompt: str = "",
        audio: Optional[bool] = None,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "input": {
                "prompt": input_prompt,
                "img_url": encoded_image,
            },
            "parameters": {
                "resolution": resolution,
                "duration": duration,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
            },
        }
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        if audio is not None:
            payload["parameters"]["audio"] = audio
        if seed is not None:
            payload["parameters"]["seed"] = seed
        payload["parameters"].update(kwargs)
        return self._post_task(payload)

    def generate_r2av(
        self,
        reference_urls: List[str],
        input_prompt: str,
        model: str = "wan2.6-r2v",
        size: str = "1280*720",
        duration: int = 5,
        negative_prompt: str = "",
        audio: Optional[bool] = None,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "input": {
                "prompt": input_prompt,
                "reference_urls": reference_urls,
            },
            "parameters": {
                "size": size,
                "duration": duration,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
            },
        }
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        if audio is not None:
            payload["parameters"]["audio"] = audio
        if seed is not None:
            payload["parameters"]["seed"] = seed
        payload["parameters"].update(kwargs)
        return self._post_task(payload)

    def get_task(self, task_id: str) -> Dict[str, Any]:
        response = requests.get(
            self._task_status_url(task_id),
            headers=self._headers(async_request=False),
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def download_video(
        self,
        video_url: str,
        save_path: str,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(video_url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with output_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
        return str(output_path)

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        model: Optional[str] = None,
        size: str = "1280*720",
        resolution: str = "720P",
        duration: int = 5,
        negative_prompt: str = "",
        audio: Optional[bool] = None,
        prompt_extend: bool = True,
        shot_type: Optional[str] = None,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        prompt = processed_data.get("prompt", "")
        encoded_image = processed_data.get("encoded_image", None)
        reference_urls = processed_data.get("reference_urls", None)
        audio_url = processed_data.get("audio_url", None)

        if task_type == "auto":
            if reference_urls:
                task_type = "r2av"
            elif encoded_image is not None:
                task_type = "i2av"
            else:
                task_type = "t2av"

        if task_type == "t2av":
            response = self.generate_t2av(
                input_prompt=prompt,
                model=model or "wan2.6-t2v",
                size=size,
                duration=duration,
                negative_prompt=negative_prompt,
                audio_url=audio_url,
                prompt_extend=prompt_extend,
                shot_type=shot_type,
                watermark=watermark,
                seed=seed,
                **kwargs,
            )
        elif task_type == "i2av":
            if encoded_image is None:
                raise ValueError("i2av task requires images input.")
            response = self.generate_i2av(
                encoded_image=encoded_image,
                input_prompt=prompt,
                model=model or "wan2.6-i2v",
                resolution=resolution,
                duration=duration,
                negative_prompt=negative_prompt,
                audio=audio,
                prompt_extend=prompt_extend,
                watermark=watermark,
                seed=seed,
                **kwargs,
            )
        elif task_type == "r2av":
            if not reference_urls:
                raise ValueError("r2av task requires reference_urls input.")
            response = self.generate_r2av(
                reference_urls=reference_urls,
                input_prompt=prompt,
                model=model or "wan2.6-r2v",
                size=size,
                duration=duration,
                negative_prompt=negative_prompt,
                audio=audio,
                prompt_extend=prompt_extend,
                watermark=watermark,
                seed=seed,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }
