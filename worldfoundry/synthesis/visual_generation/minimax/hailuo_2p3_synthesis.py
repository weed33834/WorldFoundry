"""
This module provides a client for interacting with the MiniMax Hailuo 2.3 API
for text-to-audio/video (T2AV) and image-to-audio/video (I2AV) synthesis.

It encapsulates the logic for sending requests, handling responses,
and downloading generated video content.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import requests


class Hailuo2p3Synthesis(object):
    """
    A client class for interacting with the MiniMax Hailuo 2.3 Text-to-Audio/Video (T2AV)
    and Image-to-Audio/Video (I2AV) synthesis API.

    This class handles API requests, response validation, and video file downloads.
    """

    def __init__(
        self,
        endpoint: str = "https://api.minimax.io/v1",
        api_key: str = "your_api_key",
        logger=None,
    ):
        """
        Initializes the MiniMax Hailuo 2.3 synthesis client.

        Args:
            endpoint (str): The base URL for the MiniMax API. Defaults to "https://api.minimax.io/v1".
            api_key (str): Your MiniMax API key. Defaults to "your_api_key".
            logger (Optional[Any]): An optional logger instance for logging messages.
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.logger = logger

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://api.minimax.io/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ):
        """
        A class method to initialize the MiniMax Hailuo 2.3 synthesis client.
        This serves as an alternative constructor.

        Args:
            endpoint (str): The base URL for the MiniMax API. Defaults to "https://api.minimax.io/v1".
            api_key (str): Your MiniMax API key. Defaults to "your_api_key".
            logger (Optional[Any]): An optional logger instance for logging messages.
            **kwargs: Additional keyword arguments to be passed during initialization (currently not used).

        Returns:
            Hailuo2p3Synthesis: An instance of the Hailuo2p3Synthesis client.
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
            Dict[str, str]: A dictionary containing authorization and content type headers.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, route: str) -> str:
        """
        Constructs a full API URL by joining the base endpoint with a specific route.

        Args:
            route (str): The specific API route (e.g., "/video_generation").

        Returns:
            str: The complete URL for the API request.
        """
        return f"{self.endpoint}/{route.lstrip('/')}"

    def _ensure_success(self, payload: Dict[str, Any]) -> None:
        """
        Checks the `base_resp` from an API payload to ensure the request was successful.

        Raises a RuntimeError if the status code indicates a failure.

        Args:
            payload (Dict[str, Any]): The JSON response payload from the API.

        Raises:
            RuntimeError: If the API response indicates an error (status_code is not 0).
        """
        base_resp = payload.get("base_resp", {})
        status_code = base_resp.get("status_code", 0)
        # Check if the status_code is neither 0 nor "0", indicating an error.
        if status_code not in (0, "0"):
            raise RuntimeError(
                base_resp.get("status_msg", f"MiniMax request failed with status_code={status_code}")
            )

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends a POST request to the MiniMax API and processes the response.

        Handles network errors, parses the JSON response, and performs an
        API-specific success check.

        Args:
            route (str): The API route for the POST request.
            payload (Dict[str, Any]): The JSON payload to be sent in the request body.

        Returns:
            Dict[str, Any]: The parsed JSON response from the API.

        Raises:
            requests.exceptions.RequestException: If the HTTP request itself fails.
            RuntimeError: If the API response indicates an error.
        """
        response = requests.post(
            self._url(route),
            headers=self._headers(),
            json=payload,
            timeout=300,  # Set a timeout for the request to prevent hanging.
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx).
        result = response.json()
        self._ensure_success(result)  # Check MiniMax's internal success status.
        return result

    def generate_t2av(
        self,
        input_prompt: str,
        model: str = "MiniMax-Hailuo-2.3",
        resolution: str = "768P",
        duration: int = 6,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates audio/video from a text prompt using the MiniMax Hailuo 2.3 API.

        Args:
            input_prompt (str): The text prompt to generate video from.
            model (str): The specific model to use for generation. Defaults to "MiniMax-Hailuo-2.3".
            resolution (str): The desired resolution for the generated video (e.g., "768P").
            duration (int): The desired duration of the video in seconds.
            **kwargs: Additional parameters to include in the request payload.

        Returns:
            Dict[str, Any]: The API response containing task details, including task_id.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": input_prompt,
            "resolution": resolution,
            "duration": duration,
        }
        payload.update(kwargs)  # Add any extra keyword arguments to the payload.
        return self._post("/video_generation", payload)

    def generate_i2av(
        self,
        first_frame_image: str,
        input_prompt: str,
        model: str = "MiniMax-Hailuo-2.3",
        resolution: str = "768P",
        duration: int = 6,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates audio/video from an initial image and a text prompt using the MiniMax Hailuo 2.3 API.

        Args:
            first_frame_image (str): The base64-encoded string of the first frame image.
            input_prompt (str): The text prompt to guide the video generation.
            model (str): The specific model to use for generation. Defaults to "MiniMax-Hailuo-2.3".
            resolution (str): The desired resolution for the generated video (e.g., "768P").
            duration (int): The desired duration of the video in seconds.
            **kwargs: Additional parameters to include in the request payload.

        Returns:
            Dict[str, Any]: The API response containing task details, including task_id.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": input_prompt,
            "first_frame_image": first_frame_image,
            "resolution": resolution,
            "duration": duration,
        }
        payload.update(kwargs)  # Add any extra keyword arguments to the payload.
        return self._post("/video_generation", payload)

    def query_task(self, task_id: str) -> Dict[str, Any]:
        """
        Queries the status and details of a previously submitted video generation task.

        Args:
            task_id (str): The unique identifier of the task to query.

        Returns:
            Dict[str, Any]: The API response with task status and details.
        """
        response = requests.get(
            self._url("/query/video_generation"),
            headers=self._headers(),
            params={"task_id": task_id},  # Query parameters for task ID.
            timeout=300,
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx).
        result = response.json()
        self._ensure_success(result)  # Check MiniMax's internal success status.
        return result

    def retrieve_file(self, file_id: str) -> Dict[str, Any]:
        """
        Retrieves metadata and download information for a generated file from the MiniMax API.

        Args:
            file_id (str): The unique identifier of the file to retrieve.

        Returns:
            Dict[str, Any]: The API response with file metadata and potentially a download URL.
        """
        response = requests.get(
            self._url("/files/retrieve"),
            headers=self._headers(),
            params={"file_id": file_id},  # Query parameters for file ID.
            timeout=300,
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx).
        result = response.json()
        self._ensure_success(result)  # Check MiniMax's internal success status.
        return result

    def download_video(
        self,
        video_url: str,
        save_path: str,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Downloads a video file from a given URL to a specified local path.

        This method streams the download, making it suitable for large files,
        and ensures the target directory exists.

        Args:
            video_url (str): The direct URL to the video file.
            save_path (str): The local file path where the video should be saved.
            chunk_size (int): The size of each chunk to read/write during streaming (in bytes).

        Returns:
            str: The string representation of the path where the video was saved.

        Raises:
            requests.exceptions.RequestException: If the HTTP request to download the video fails.
            IOError: If there is an issue writing the file to disk.
        """
        output_path = Path(save_path)
        # Ensure the parent directory for the save_path exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(video_url, stream=True, timeout=300) as response:
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx).
            with output_path.open("wb") as file:
                # Iterate over the content in chunks to efficiently write large files.
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks.
                        file.write(chunk)
        return str(output_path)

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        model: str = "MiniMax-Hailuo-2.3",
        resolution: str = "768P",
        duration: int = 6,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Orchestrates video generation based on input data, automatically determining
        the task type (T2AV or I2AV) if not explicitly specified.

        Args:
            processed_data (Dict[str, Any]): A dictionary containing input data, expected to have
                                             "prompt" and optionally "first_frame_image".
            task_type (str): The type of generation task: "auto", "t2av" (text-to-video),
                             or "i2av" (image-to-video). "auto" infers based on `processed_data`.
            model (str): The specific model to use for generation. Defaults to "MiniMax-Hailuo-2.3".
            resolution (str): The desired resolution for the generated video (e.g., "768P").
            duration (int): The desired duration of the video in seconds.
            **kwargs: Additional parameters to pass to the underlying generation function.

        Returns:
            Dict[str, Any]: A dictionary containing the determined task type, the prompt used,
                            and the raw API response from the generation call.

        Raises:
            ValueError: If an unsupported `task_type` is provided or if `i2av` is
                        selected without a `first_frame_image`.
        """
        prompt = processed_data.get("prompt", "")
        first_frame_image = processed_data.get("first_frame_image", None)

        # Automatically determine task type based on the presence of first_frame_image.
        if task_type == "auto":
            if first_frame_image is not None:
                task_type = "i2av"
            else:
                task_type = "t2av"

        if task_type == "t2av":
            response = self.generate_t2av(
                input_prompt=prompt,
                model=model,
                resolution=resolution,
                duration=duration,
                **kwargs,
            )
        elif task_type == "i2av":
            # Validate that an image is provided if i2av task type is explicitly chosen or inferred.
            if first_frame_image is None:
                raise ValueError("i2av task requires images input.")
            response = self.generate_i2av(
                first_frame_image=first_frame_image,
                input_prompt=prompt,
                model=model,
                resolution=resolution,
                duration=duration,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }