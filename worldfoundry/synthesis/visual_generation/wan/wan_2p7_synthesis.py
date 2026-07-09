"""
This module provides a client for interacting with the Wan2.7 API for video synthesis (Image-to-Audio/Video).

It includes functionalities to generate video tasks, query task status, and download generated videos.
"""

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


class Wan2p7Synthesis(object):
    """
    Client for interacting with the Wan2.7 API for video synthesis.

    This class provides methods to generate videos from images and prompts,
    check the status of synthesis tasks, and download the resulting videos.
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
        """
        Initializes the Wan2p7Synthesis client.

        Args:
            endpoint (str): The base URL for the Wan2.7 API.
                            Defaults to "https://dashscope.aliyuncs.com/api/v1".
            api_key (str): Your API key for authentication. Defaults to "your_api_key".
            logger: An optional logger object for logging messages.
        """
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
        """
        Class method to initialize the Wan2p7Synthesis client,
        primarily used for consistency with other API clients.

        Args:
            endpoint (str): The base URL for the Wan2.7 API.
                            Defaults to "https://dashscope.aliyuncs.com/api/v1".
            api_key (str): Your API key for authentication. Defaults to "your_api_key".
            logger: An optional logger object for logging messages.
            **kwargs: Additional keyword arguments (currently not used).

        Returns:
            Wan2p7Synthesis: An instance of the Wan2p7Synthesis client.
        """
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
        )

    def _headers(self, async_request: bool = True) -> Dict[str, str]:
        """
        Constructs the HTTP headers required for API requests.

        Args:
            async_request (bool): If True, adds the 'X-DashScope-Async' header to enable
                                  asynchronous task processing. Defaults to True.

        Returns:
            Dict[str, str]: A dictionary of HTTP headers.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if async_request:
            # Enable asynchronous processing for the request
            headers["X-DashScope-Async"] = "enable"
        return headers

    def _create_url(self) -> str:
        """
        Constructs the full URL for the video synthesis task creation endpoint.

        Returns:
            str: The complete URL for creating a synthesis task.
        """
        task_path = "/services/aigc/video-generation/video-synthesis"
        # If the endpoint already contains the task_path, return it directly.
        # Otherwise, append the task_path to the endpoint.
        if self.endpoint.endswith(task_path):
            return self.endpoint
        return f"{self.endpoint}{task_path}"

    def _task_status_url(self, task_id: str) -> str:
        """
        Constructs the full URL for querying the status of a specific task.

        Args:
            task_id (str): The ID of the synthesis task.

        Returns:
            str: The complete URL for checking task status.
        """
        task_path = "/services/aigc/video-generation/video-synthesis"
        base_endpoint = self.endpoint
        # Remove the task_path from the base endpoint if it's present,
        # to correctly form the task status URL.
        if base_endpoint.endswith(task_path):
            base_endpoint = base_endpoint[:-len(task_path)]
        return f"{base_endpoint}/tasks/{task_id}"

    def _post_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends a POST request to create a new synthesis task.

        Args:
            payload (Dict[str, Any]): The request body containing task parameters.

        Returns:
            Dict[str, Any]: The JSON response from the API, typically containing task ID.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails.
        """
        response = requests.post(
            self._create_url(),
            headers=self._headers(async_request=True),
            json=payload,
            timeout=300,
        )
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()

    def generate_i2av(
        self,
        media: List[Dict[str, str]],
        input_prompt: str,
        model: str = "wan2.7-i2v",
        resolution: str = "720P",
        duration: int = 5,
        negative_prompt: str = "",
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates an Image-to-Audio/Video (i2av) synthesis task.

        Args:
            media (List[Dict[str, str]]): A list of media inputs, where each dict
                                         specifies 'type' (e.g., "image") and 'url'.
            input_prompt (str): The main prompt describing the desired video content.
            model (str): The model to use for synthesis. Defaults to "wan2.7-i2v".
            resolution (str): The desired video resolution (e.g., "720P"). Defaults to "720P".
            duration (int): The desired video duration in seconds. Defaults to 5.
            negative_prompt (str): A prompt describing what should NOT be in the video.
                                   Defaults to "".
            prompt_extend (bool): Whether to extend the prompt internally. Defaults to True.
            watermark (bool): Whether to add a watermark to the video. Defaults to False.
            seed (Optional[int]): The random seed for reproducibility. Defaults to None.
            **kwargs: Additional parameters to be passed directly to the 'parameters' field
                      of the API request payload.

        Returns:
            Dict[str, Any]: The JSON response from the API, typically including the task ID.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "input": {
                "prompt": input_prompt,
                "media": media,
            },
            "parameters": {
                "resolution": resolution,
                "duration": duration,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
            },
        }
        # Conditionally add optional parameters to the payload
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["parameters"]["seed"] = seed
        # Add any extra keyword arguments to the parameters section of the payload
        payload["parameters"].update(kwargs)
        return self._post_task(payload)

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """
        Retrieves the status and results of a previously submitted synthesis task.

        Args:
            task_id (str): The ID of the task to query.

        Returns:
            Dict[str, Any]: The JSON response from the API, containing task status
                            and potentially output URLs.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails.
        """
        response = requests.get(
            self._task_status_url(task_id),
            headers=self._headers(async_request=False),  # Task status check is typically synchronous
            timeout=300,
        )
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()

    def download_video(
        self,
        video_url: str,
        save_path: str,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Downloads a video from a given URL and saves it to a specified path.

        Args:
            video_url (str): The URL of the video to download.
            save_path (str): The local file path where the video will be saved.
            chunk_size (int): The size of chunks to read/write during download in bytes.
                              Defaults to 1MB.

        Returns:
            str: The absolute path to the downloaded video file.

        Raises:
            requests.exceptions.RequestException: If the HTTP request for download fails.
            IOError: If there's an issue writing the file to disk.
        """
        output_path = Path(save_path)
        # Ensure the parent directory for the save_path exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(video_url, stream=True, timeout=300) as response:
            response.raise_for_status()  # Check for HTTP errors
            with output_path.open("wb") as file:
                # Iterate over the content in chunks to handle large files efficiently
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive new chunks
                        file.write(chunk)
        return str(output_path)

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "i2av",
        model: Optional[str] = None,
        resolution: str = "720P",
        duration: int = 5,
        negative_prompt: str = "",
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Submits a video synthesis prediction task based on processed input data.

        This method acts as a unified interface for various synthesis tasks,
        though currently only 'i2av' is supported for Wan2.7.

        Args:
            processed_data (Dict[str, Any]): A dictionary containing input data for the task.
                                             Must include 'prompt' (str) and 'media' (List[Dict[str, str]]).
            task_type (str): The type of synthesis task. Currently, only "i2av" is supported.
                             Defaults to "i2av".
            model (Optional[str]): The model to use for synthesis. If None, defaults to "wan2.7-i2v".
            resolution (str): The desired video resolution (e.g., "720P"). Defaults to "720P".
            duration (int): The desired video duration in seconds. Defaults to 5.
            negative_prompt (str): A prompt describing what should NOT be in the video.
                                   Defaults to "".
            prompt_extend (bool): Whether to extend the prompt internally. Defaults to True.
            watermark (bool): Whether to add a watermark to the video. Defaults to False.
            seed (Optional[int]): The random seed for reproducibility. Defaults to None.
            **kwargs: Additional parameters to be passed directly to the `generate_i2av` method.

        Returns:
            Dict[str, Any]: A dictionary containing the task type, input prompt, and the API
                            response for the created task.

        Raises:
            ValueError: If an unsupported `task_type` is provided or if `media` input is missing.
        """
        if task_type != "i2av":
            raise ValueError("Wan2.7 currently only supports i2av task_type.")

        prompt = processed_data.get("prompt", "")
        media = processed_data.get("media", None)
        if not media:
            raise ValueError("Wan2.7 requires media input.")

        response = self.generate_i2av(
            media=media,
            input_prompt=prompt,
            model=model or "wan2.7-i2v",  # Use provided model or default
            resolution=resolution,
            duration=duration,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs,
        )

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }