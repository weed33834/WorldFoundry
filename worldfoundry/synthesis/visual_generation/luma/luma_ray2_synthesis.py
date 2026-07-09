"""
Client for interacting with the Luma Labs Dream Machine API, specifically for generating and managing video synthesis.

This module provides a Python class `LumaRay2Synthesis` to facilitate text-to-video (T2AV)
and image-to-video (I2AV) generation, as well as checking generation status and downloading results.
"""
from pathlib import Path
from typing import Optional, Dict, Any

import requests


class LumaRay2Synthesis(object):
    """
    Client for interacting with the Luma Labs Dream Machine API (Ray-2 Synthesis).

    This class provides methods to generate videos from text prompts or keyframes,
    retrieve generation status, and download synthesized videos.
    """

    def __init__(
        self,
        endpoint: str = "https://api.lumalabs.ai/dream-machine/v1",
        api_key: str = "your_api_key",
        logger=None,
    ):
        """
        Initializes the LumaRay2Synthesis client.

        Args:
            endpoint (str): The base URL for the Luma Labs Dream Machine API.
                            Defaults to "https://api.lumalabs.ai/dream-machine/v1".
            api_key (str): Your Luma Labs API key. Defaults to "your_api_key".
            logger (Any, optional): An optional logger object for logging messages.
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.logger = logger

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://api.lumalabs.ai/dream-machine/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ):
        """
        Factory method to initialize the LumaRay2Synthesis client.

        This method acts as an alternative constructor, allowing for future expansion
        with additional keyword arguments if needed, while maintaining a clear
        interface for basic initialization.

        Args:
            endpoint (str): The base URL for the Luma Labs Dream Machine API.
                            Defaults to "https://api.lumalabs.ai/dream-machine/v1".
            api_key (str): Your Luma Labs API key. Defaults to "your_api_key".
            logger (Any, optional): An optional logger object for logging messages.
            **kwargs: Additional keyword arguments (currently unused but reserved for future compatibility).

        Returns:
            LumaRay2Synthesis: An instance of the LumaRay2Synthesis client.
        """
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
        )

    def _headers(self) -> Dict[str, str]:
        """
        Constructs the standard HTTP headers for API requests.

        Returns:
            Dict[str, str]: A dictionary containing Authorization and Content-Type headers.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, route: str) -> str:
        """
        Constructs a full API URL from a given route.

        Args:
            route (str): The specific API endpoint route (e.g., "/generations").

        Returns:
            str: The complete URL for the API request.
        """
        # Ensure the route is prefixed with a single slash and combined with the base endpoint.
        return f"{self.endpoint}/{route.lstrip('/')}"

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends a POST request to the Luma Labs API.

        Args:
            route (str): The API route for the POST request (e.g., "/generations").
            payload (Dict[str, Any]): The JSON payload to be sent in the request body.

        Returns:
            Dict[str, Any]: The JSON response from the API.

        Raises:
            requests.exceptions.RequestException: If the POST request fails or returns a non-2xx status code.
        """
        response = requests.post(
            self._url(route),
            headers=self._headers(),
            json=payload,
            timeout=300,  # Set a timeout for the request to prevent indefinite hanging.
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx).
        return response.json()

    def generate_t2av(
        self,
        input_prompt: str,
        model: str = "ray-2",
        resolution: str = "720p",
        duration: str = "5s",
        aspect_ratio: Optional[str] = "16:9",
        loop: bool = False,
        concepts: Optional[Any] = None,
        callback_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates an audio/video clip from a text prompt (Text-to-Audio/Video).

        Args:
            input_prompt (str): The text prompt to generate the video from.
            model (str): The generation model to use (e.g., "ray-2"). Defaults to "ray-2".
            resolution (str): The desired video resolution (e.g., "720p", "1080p"). Defaults to "720p".
            duration (str): The desired video duration (e.g., "5s", "10s"). Defaults to "5s".
            aspect_ratio (Optional[str]): The aspect ratio of the video (e.g., "16:9", "1:1"). Defaults to "16:9".
            loop (bool): Whether the generated video should loop. Defaults to False.
            concepts (Optional[Any]): Additional concepts or embeddings to guide generation. Defaults to None.
            callback_url (Optional[str]): An optional URL to receive callbacks upon generation completion. Defaults to None.
            **kwargs: Additional parameters to include in the request payload.

        Returns:
            Dict[str, Any]: The API response containing details about the initiated generation,
                            typically including a generation ID.
        """
        payload: Dict[str, Any] = {
            "prompt": input_prompt,
            "model": model,
            "resolution": resolution,
            "duration": duration,
            "loop": loop,
        }
        # Conditionally add optional parameters to the payload if they are provided.
        if aspect_ratio is not None:
            payload["aspect_ratio"] = aspect_ratio
        if concepts is not None:
            payload["concepts"] = concepts
        if callback_url is not None:
            payload["callback_url"] = callback_url
        payload.update(kwargs)  # Add any extra kwargs directly to the payload.
        return self._post("/generations", payload)

    def generate_i2av(
        self,
        keyframes: Dict[str, Any],
        input_prompt: str,
        model: str = "ray-2",
        resolution: str = "720p",
        duration: str = "5s",
        aspect_ratio: Optional[str] = "16:9",
        loop: bool = False,
        concepts: Optional[Any] = None,
        callback_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates an audio/video clip from a text prompt and keyframes (Image-to-Audio/Video).

        Args:
            keyframes (Dict[str, Any]): A dictionary describing the keyframes for the video generation.
                                       This typically includes frame data and timestamps.
            input_prompt (str): The text prompt to guide the video generation.
            model (str): The generation model to use (e.g., "ray-2"). Defaults to "ray-2".
            resolution (str): The desired video resolution (e.g., "720p", "1080p"). Defaults to "720p".
            duration (str): The desired video duration (e.g., "5s", "10s"). Defaults to "5s".
            aspect_ratio (Optional[str]): The aspect ratio of the video (e.g., "16:9", "1:1"). Defaults to "16:9".
            loop (bool): Whether the generated video should loop. Defaults to False.
            concepts (Optional[Any]): Additional concepts or embeddings to guide generation. Defaults to None.
            callback_url (Optional[str]): An optional URL to receive callbacks upon generation completion. Defaults to None.
            **kwargs: Additional parameters to include in the request payload.

        Returns:
            Dict[str, Any]: The API response containing details about the initiated generation,
                            typically including a generation ID.
        """
        payload: Dict[str, Any] = {
            "prompt": input_prompt,
            "model": model,
            "resolution": resolution,
            "duration": duration,
            "loop": loop,
            "keyframes": keyframes,
        }
        # Conditionally add optional parameters to the payload if they are provided.
        if aspect_ratio is not None:
            payload["aspect_ratio"] = aspect_ratio
        if concepts is not None:
            payload["concepts"] = concepts
        if callback_url is not None:
            payload["callback_url"] = callback_url
        payload.update(kwargs)  # Add any extra kwargs directly to the payload.
        return self._post("/generations", payload)

    def get_generation(self, generation_id: str) -> Dict[str, Any]:
        """
        Retrieves the status and details of a specific video generation task.

        Args:
            generation_id (str): The unique identifier of the generation task.

        Returns:
            Dict[str, Any]: The API response containing the generation status, progress, and output details.

        Raises:
            requests.exceptions.RequestException: If the GET request fails or returns a non-2xx status code.
        """
        response = requests.get(
            self._url(f"/generations/{generation_id}"),
            headers=self._headers(),
            timeout=300,  # Set a timeout for the request.
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx).
        return response.json()

    def download_video(
        self,
        video_url: str,
        save_path: str,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Downloads a video from a given URL and saves it to a specified local path.

        Args:
            video_url (str): The direct URL to the video file.
            save_path (str): The local file path where the video should be saved.
            chunk_size (int): The size of chunks (in bytes) to read from the stream.
                              Defaults to 1MB (1024 * 1024).

        Returns:
            str: The absolute path to the downloaded video file.

        Raises:
            requests.exceptions.RequestException: If the GET request for the video fails or returns a non-2xx status code.
        """
        output_path = Path(save_path)
        # Create parent directories if they don't exist, ensuring the save_path is valid.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(video_url, stream=True, timeout=300) as response:
            response.raise_for_status()  # Ensure the download request was successful.
            with output_path.open("wb") as file:
                # Iterate over the response content in chunks to handle large files efficiently.
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks.
                        file.write(chunk)
        return str(output_path)

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        model: str = "ray-2",
        resolution: str = "720p",
        duration: str = "5s",
        aspect_ratio: Optional[str] = "16:9",
        loop: bool = False,
        concepts: Optional[Any] = None,
        callback_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        A unified interface to trigger video generation based on the input data and task type.

        This method automatically determines if a text-to-video (T2AV) or image-to-video (I2AV)
        task should be performed based on the presence of 'keyframes' in `processed_data`,
        unless `task_type` is explicitly specified.

        Args:
            processed_data (Dict[str, Any]): A dictionary containing input data,
                                              e.g., {"prompt": "...", "keyframes": {...}}.
            task_type (str): The type of generation task: "auto", "t2av", or "i2av".
                             If "auto", it infers from `processed_data`. Defaults to "auto".
            model (str): The generation model to use. Defaults to "ray-2".
            resolution (str): The desired video resolution. Defaults to "720p".
            duration (str): The desired video duration. Defaults to "5s".
            aspect_ratio (Optional[str]): The aspect ratio of the video. Defaults to "16:9".
            loop (bool): Whether the generated video should loop. Defaults to False.
            concepts (Optional[Any]): Additional concepts or embeddings. Defaults to None.
            callback_url (Optional[str]): An optional URL for callbacks. Defaults to None.
            **kwargs: Additional parameters passed to the underlying generation methods.

        Returns:
            Dict[str, Any]: A dictionary containing the determined task type, the prompt used,
                            and the raw API response from the generation call.

        Raises:
            ValueError: If an unsupported `task_type` is provided or if an "i2av" task
                        is requested without `keyframes`.
        """
        prompt = processed_data.get("prompt", "")
        keyframes = processed_data.get("keyframes", None)

        # Automatically determine the task type based on the presence of keyframes if 'auto' is selected.
        if task_type == "auto":
            if keyframes:
                task_type = "i2av"
            else:
                task_type = "t2av"

        if task_type == "t2av":
            response = self.generate_t2av(
                input_prompt=prompt,
                model=model,
                resolution=resolution,
                duration=duration,
                aspect_ratio=aspect_ratio,
                loop=loop,
                concepts=concepts,
                callback_url=callback_url,
                **kwargs,
            )
        elif task_type == "i2av":
            # Ensure keyframes are provided for an image-to-audio/video task.
            if not keyframes:
                raise ValueError("i2av task requires keyframes input.")
            response = self.generate_i2av(
                keyframes=keyframes,
                input_prompt=prompt,
                model=model,
                resolution=resolution,
                duration=duration,
                aspect_ratio=aspect_ratio,
                loop=loop,
                concepts=concepts,
                callback_url=callback_url,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }