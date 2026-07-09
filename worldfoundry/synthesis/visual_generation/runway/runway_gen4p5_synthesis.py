"""
This module provides a client for interacting with the Runway Gen-4.5 API for video synthesis tasks.
It allows for generating videos from text or images, checking task status, and downloading generated videos.
"""

from pathlib import Path
from typing import Optional, Dict, Any


class RunwayGen4p5Synthesis(object):
    """
    Client for synthesizing videos using the Runway Gen-4.5 API.

    This class provides methods to interact with the Runway API,
    allowing users to generate videos from text prompts (text-to-video)
    or from image prompts combined with text (image-to-video),
    retrieve task information, and download the resulting videos.
    """

    def __init__(
        self,
        endpoint: str = "https://api.dev.runwayml.com/v1",
        api_key: str = "your_api_key",
        runway_version: str = "2024-11-06",
        logger=None,
    ):
        """
        Initializes the Runway Gen-4.5 API synthesis client.

        Args:
            endpoint (str): The base URL for the Runway API. Defaults to "https://api.dev.runwayml.com/v1".
            api_key (str): Your Runway API key. Defaults to "your_api_key".
            runway_version (str): The target Runway API version. Defaults to "2024-11-06".
            logger: An optional logger object to use for logging messages.
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.runway_version = runway_version
        self.logger = logger

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://api.dev.runwayml.com/v1",
        api_key: str = "your_api_key",
        runway_version: str = "2024-11-06",
        logger=None,
        **kwargs
    ):
        """
        Factory method to initialize the Runway Gen-4.5 API synthesis client.

        This class method allows for alternative initialization patterns,
        passing all arguments directly to the constructor.

        Args:
            endpoint (str): The base URL for the Runway API. Defaults to "https://api.dev.runwayml.com/v1".
            api_key (str): Your Runway API key. Defaults to "your_api_key".
            runway_version (str): The target Runway API version. Defaults to "2024-11-06".
            logger: An optional logger object to use for logging messages.
            **kwargs: Additional keyword arguments to pass to the constructor (currently not used explicitly).

        Returns:
            RunwayGen4p5Synthesis: An instance of the RunwayGen4p5Synthesis class.
        """
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            runway_version=runway_version,
            logger=logger,
        )

    def _headers(self) -> Dict[str, str]:
        """
        Constructs the standard HTTP headers required for Runway API requests.

        Returns:
            Dict[str, str]: A dictionary containing authorization, content-type,
                            and Runway version headers.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": self.runway_version,
        }

    def _url(self, route: str) -> str:
        """
        Constructs the full URL for an API endpoint.

        Args:
            route (str): The specific API route (e.g., "/text_to_video").

        Returns:
            str: The complete URL by combining the base endpoint and the route.
        """
        return f"{self.endpoint}/{route.lstrip('/')}"

    def _requests(self):
        """
        Lazily imports and returns the 'requests' library.

        This approach defers the import of the 'requests' library until it's
        actually needed, which can be useful in environments where not all
        dependencies are always available or to speed up initial module load.

        Returns:
            module: The imported 'requests' module.
        """
        import requests
        return requests

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends a POST request to a specified Runway API route.

        Args:
            route (str): The API route for the POST request.
            payload (Dict[str, Any]): The JSON payload to send in the request body.

        Returns:
            Dict[str, Any]: The JSON response from the API.

        Raises:
            requests.exceptions.HTTPError: If the HTTP request returns an unsuccessful status code.
        """
        response = self._requests().post(
            self._url(route),
            headers=self._headers(),
            json=payload,
            timeout=300,
        )
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()

    def generate_t2av(
        self,
        input_prompt: str,
        model: str = "gen4.5",
        ratio: str = "1280:720",
        duration: int = 5,
        seed: Optional[int] = None,
        public_figure_threshold: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates a video from a text prompt (Text-to-Audio-Video).

        Args:
            input_prompt (str): The text prompt to generate the video from.
            model (str): The name of the generation model to use. Defaults to "gen4.5".
            ratio (str): The aspect ratio of the generated video (e.g., "1280:720"). Defaults to "1280:720".
            duration (int): The desired duration of the video in seconds. Defaults to 5.
            seed (Optional[int]): An optional seed for reproducible generation.
            public_figure_threshold (Optional[str]): Content moderation setting for public figures.
                                                    e.g., "high", "medium", "low", "none".
            **kwargs: Additional parameters to include in the request payload.

        Returns:
            Dict[str, Any]: The API response, typically containing task details
                            like a task ID and status.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "promptText": input_prompt,
            "ratio": ratio,
            "duration": duration,
        }
        # Conditionally add optional parameters to the payload if they are provided.
        if seed is not None:
            payload["seed"] = seed
        if public_figure_threshold is not None:
            payload["contentModeration"] = {
                "publicFigureThreshold": public_figure_threshold,
            }
        payload.update(kwargs)  # Merge any additional keyword arguments into the payload.
        return self._post("/text_to_video", payload)

    def generate_i2av(
        self,
        prompt_image: Any,
        input_prompt: str,
        model: str = "gen4.5",
        ratio: str = "1280:720",
        duration: int = 5,
        seed: Optional[int] = None,
        public_figure_threshold: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates a video from an image prompt and a text prompt (Image-to-Audio-Video).

        Args:
            prompt_image (Any): The image input for the prompt. This typically would be
                                a base64 encoded string or a URL to an image.
            input_prompt (str): The text prompt to guide the video generation.
            model (str): The name of the generation model to use. Defaults to "gen4.5".
            ratio (str): The aspect ratio of the generated video (e.g., "1280:720"). Defaults to "1280:720".
            duration (int): The desired duration of the video in seconds. Defaults to 5.
            seed (Optional[int]): An optional seed for reproducible generation.
            public_figure_threshold (Optional[str]): Content moderation setting for public figures.
                                                    e.g., "high", "medium", "low", "none".
            **kwargs: Additional parameters to include in the request payload.

        Returns:
            Dict[str, Any]: The API response, typically containing task details
                            like a task ID and status.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "promptText": input_prompt,
            "promptImage": prompt_image,
            "ratio": ratio,
            "duration": duration,
        }
        # Conditionally add optional parameters to the payload if they are provided.
        if seed is not None:
            payload["seed"] = seed
        if public_figure_threshold is not None:
            payload["contentModeration"] = {
                "publicFigureThreshold": public_figure_threshold,
            }
        payload.update(kwargs)  # Merge any additional keyword arguments into the payload.
        return self._post("/image_to_video", payload)

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """
        Retrieves the status and details of a specific video generation task.

        Args:
            task_id (str): The ID of the task to retrieve.

        Returns:
            Dict[str, Any]: The API response containing the task's current status and details.

        Raises:
            requests.exceptions.HTTPError: If the HTTP request returns an unsuccessful status code.
        """
        response = self._requests().get(
            self._url(f"/tasks/{task_id}"),
            headers=self._headers(),
            timeout=300,
        )
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
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
            save_path (str): The local file path where the video should be saved.
            chunk_size (int): The size of chunks (in bytes) to read from the stream
                              during download. Defaults to 1MB.

        Returns:
            str: The absolute path to the downloaded video file.

        Raises:
            requests.exceptions.HTTPError: If the HTTP request for the video download fails.
        """
        output_path = Path(save_path)
        # Ensure the directory for the output file exists, creating parents if necessary.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream the download to handle potentially large video files efficiently.
        with self._requests().get(video_url, stream=True, timeout=300) as response:
            response.raise_for_status()  # Check if the download request itself was successful.
            # Open the local file in binary write mode.
            with output_path.open("wb") as file:
                # Iterate over content in chunks to write to file.
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        file.write(chunk)
        return str(output_path)

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        model: str = "gen4.5",
        ratio: str = "1280:720",
        duration: int = 5,
        seed: Optional[int] = None,
        public_figure_threshold: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initiates a video generation task based on processed input data.

        Automatically determines whether to perform text-to-video (t2av)
        or image-to-video (i2av) based on the presence of an image prompt
        in the `processed_data`, unless `task_type` is explicitly specified.

        Args:
            processed_data (Dict[str, Any]): A dictionary containing input data,
                                            expected to have 'prompt' (str) and
                                            optionally 'prompt_image' (Any).
            task_type (str): The type of task to perform. Can be "auto", "t2av", or "i2av".
                            If "auto", it infers from `processed_data`. Defaults to "auto".
            model (str): The name of the generation model to use. Defaults to "gen4.5".
            ratio (str): The aspect ratio of the generated video. Defaults to "1280:720".
            duration (int): The desired duration of the video in seconds. Defaults to 5.
            seed (Optional[int]): An optional seed for reproducible generation.
            public_figure_threshold (Optional[str]): Content moderation setting for public figures.
            **kwargs: Additional parameters to pass to the underlying generation functions.

        Returns:
            Dict[str, Any]: A dictionary containing the determined task type,
                            the prompt used, and the raw API response from
                            the generation call.

        Raises:
            ValueError: If an "i2av" task is specified but no 'prompt_image' is provided,
                        or if an unsupported `task_type` is given.
        """
        prompt = processed_data.get("prompt", "")
        prompt_image = processed_data.get("prompt_image", None)

        # Automatically determine task type based on the presence of a prompt image.
        if task_type == "auto":
            if prompt_image is not None:
                task_type = "i2av"  # If image is present, assume Image-to-Video.
            else:
                task_type = "t2av"  # Otherwise, assume Text-to-Video.

        if task_type == "t2av":
            response = self.generate_t2av(
                input_prompt=prompt,
                model=model,
                ratio=ratio,
                duration=duration,
                seed=seed,
                public_figure_threshold=public_figure_threshold,
                **kwargs,
            )
        elif task_type == "i2av":
            # Ensure an image is provided if the task type is explicitly i2av.
            if prompt_image is None:
                raise ValueError("i2av task requires images input.")
            response = self.generate_i2av(
                prompt_image=prompt_image,
                input_prompt=prompt,
                model=model,
                ratio=ratio,
                duration=duration,
                seed=seed,
                public_figure_threshold=public_figure_threshold,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }