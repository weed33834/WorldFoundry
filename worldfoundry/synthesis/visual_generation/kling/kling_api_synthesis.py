from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import requests


class KlingApiSynthesis(object):
    """
    Kling API 合成类。

    默认对接 `https://api.klingapi.com` 这一套通用 Kling 网关接口。
    """

    def __init__(
        self,
        endpoint: str = "https://api.klingapi.com",
        api_key: str = "your_api_key",
        logger=None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.logger = logger

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://api.klingapi.com",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ):
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, route: str) -> str:
        return f"{self.endpoint}/{route.lstrip('/')}"

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            self._url(route),
            headers=self._headers(),
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def generate_t2av(
        self,
        input_prompt: str,
        model: str = "kling-v2.6-pro",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        mode: str = "professional",
        negative_prompt: Optional[str] = None,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": input_prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id
        payload.update(kwargs)
        return self._post("/v1/videos/text2video", payload)

    def generate_i2av(
        self,
        image_payload: Dict[str, Any],
        input_prompt: str,
        model: str = "kling-v2.6-pro",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        mode: str = "professional",
        negative_prompt: Optional[str] = None,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": input_prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
        }
        payload[image_payload["field"]] = image_payload["value"]
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id
        payload.update(kwargs)
        return self._post("/v1/videos/image2video", payload)

    def get_task(self, task_id: str) -> Dict[str, Any]:
        response = requests.get(
            self._url(f"/v1/videos/{task_id}"),
            headers=self._headers(),
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
        model: str = "kling-v2.6-pro",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        mode: str = "professional",
        negative_prompt: Optional[str] = None,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        prompt = processed_data.get("prompt", "")
        image_payload = processed_data.get("image_payload")

        if task_type == "auto":
            task_type = "i2av" if image_payload is not None else "t2av"

        if task_type == "t2av":
            response = self.generate_t2av(
                input_prompt=prompt,
                model=model,
                duration=duration,
                aspect_ratio=aspect_ratio,
                mode=mode,
                negative_prompt=negative_prompt,
                callback_url=callback_url,
                external_task_id=external_task_id,
                **kwargs,
            )
        elif task_type == "i2av":
            if image_payload is None:
                raise ValueError("i2av task requires image input.")
            response = self.generate_i2av(
                image_payload=image_payload,
                input_prompt=prompt,
                model=model,
                duration=duration,
                aspect_ratio=aspect_ratio,
                mode=mode,
                negative_prompt=negative_prompt,
                callback_url=callback_url,
                external_task_id=external_task_id,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }
