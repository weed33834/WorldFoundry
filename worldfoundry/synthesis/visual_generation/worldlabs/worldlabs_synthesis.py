"""
Client for interacting with the World Labs / Marble API for content synthesis and management.

This module provides a Python class `WorldLabsSynthesis` to facilitate communication with the
World Labs API. It includes methods for uploading media assets, generating 3D worlds
based on prompts, and retrieving details about operations and generated worlds.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional


class WorldLabsSynthesis(object):
    """
    A client for interacting with the World Labs / Marble API.

    This class provides methods to perform various operations such as uploading media assets,
    generating 3D worlds, and fetching operation statuses or world details. It handles
    API authentication and standard request patterns.
    """

    def __init__(
        self,
        endpoint: str = "https://api.worldlabs.ai",
        api_key: str = "your_api_key",
        logger=None,
    ):
        """
        Initializes the WorldLabsSynthesis client.

        Args:
            endpoint: The base URL for the World Labs API. Defaults to "https://api.worldlabs.ai".
            api_key: Your World Labs API key. Defaults to "your_api_key".
            logger: An optional logger instance to use for logging.
        """
        # Ensure the endpoint does not end with a slash to prevent double slashes in URLs.
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.logger = logger

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://api.worldlabs.ai",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ):
        """
        An alternative class method for initializing the WorldLabsSynthesis client.

        This method provides an alternative way to instantiate the client,
        allowing for potential future expansion with additional keyword arguments.

        Args:
            endpoint: The base URL for the World Labs API. Defaults to "https://api.worldlabs.ai".
            api_key: Your World Labs API key. Defaults to "your_api_key".
            logger: An optional logger instance to use for logging.
            **kwargs: Additional keyword arguments (currently unused but reserved for future compatibility).

        Returns:
            An instance of the WorldLabsSynthesis class.
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
            A dictionary containing the API key and content type headers.
        """
        return {
            "WLT-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def _url(self, route: str) -> str:
        """
        Constructs a full API URL by combining the base endpoint and a given route.

        Args:
            route: The specific API route (e.g., "/marble/v1/worlds:generate").

        Returns:
            The complete URL for the API request.
        """
        # Ensure the route does not start with a slash to prevent double slashes in URLs.
        return f"{self.endpoint}/{route.lstrip('/')}"

    def _requests(self):
        """
        Imports the `requests` HTTP client library only when an API request is made.

        This allows for lazy loading of the `requests` library, which can be beneficial
        if the client is instantiated but not always used for making actual requests.

        Returns:
            The `requests` module.
        """
        import requests

        return requests

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends a POST request to a specified API route with a JSON payload.

        Handles common API request patterns including URL construction, headers,
        and error checking.

        Args:
            route: The API route for the POST request.
            payload: A dictionary representing the JSON body of the request.

        Returns:
            A dictionary containing the JSON response from the API.

        Raises:
            requests.exceptions.HTTPError: If the API response status code indicates an error.
        """
        response = self._requests().post(
            self._url(route),
            headers=self._headers(),
            json=payload,
            timeout=300,  # Set a timeout for the request in seconds.
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()

    def prepare_upload(
        self,
        file_name: str,
        kind: str,
        extension: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepares an upload for a media asset by requesting a pre-signed URL from the API.

        This is the first step in the media asset upload process. The API returns information
        needed to perform the actual file upload, including an `upload_url`.

        Args:
            file_name: The name of the file to be uploaded.
            kind: The type of media asset (e.g., "IMAGE", "AUDIO", "3D_MODEL").
            extension: The file extension (e.g., "png", "glb"). Optional, will be inferred if not provided.
            metadata: Optional dictionary of arbitrary metadata to associate with the asset.

        Returns:
            A dictionary containing the upload information, including the `upload_url`
            and a `media_asset` object with a `media_asset_id`.
        """
        payload: Dict[str, Any] = {
            "file_name": file_name,
            "kind": kind,
        }
        # Add optional fields to the payload if they are provided.
        if extension is not None:
            payload["extension"] = extension
        if metadata is not None:
            payload["metadata"] = metadata
        return self._post("/marble/v1/media-assets:prepare_upload", payload)

    def upload_file(
        self,
        file_path: str,
        upload_url: str,
        required_headers: Optional[Dict[str, str]] = None,
        method: str = "PUT",
    ) -> None:
        """
        Uploads a local file to a given pre-signed URL.

        This method performs the actual file transfer using the `upload_url` and
        `required_headers` obtained from the `prepare_upload` method.

        Args:
            file_path: The local path to the file to be uploaded.
            upload_url: The pre-signed URL to which the file should be uploaded.
            required_headers: Optional dictionary of HTTP headers required for the upload (e.g., authorization).
            method: The HTTP method to use for the upload (typically "PUT").

        Raises:
            requests.exceptions.HTTPError: If the upload response status code indicates an error.
            FileNotFoundError: If the specified `file_path` does not exist.
        """
        path = Path(file_path)
        # Guess the MIME type of the file based on its extension.
        # Default to 'application/octet-stream' if the type cannot be guessed.
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

        # Initialize headers with any required headers from the prepare_upload response,
        # then set or override the 'Content-Type'.
        headers = dict(required_headers or {})
        headers["Content-Type"] = content_type

        with path.open("rb") as file:
            response = self._requests().request(
                method=method or "PUT",  # Use provided method or default to PUT
                url=upload_url,
                headers=headers,
                data=file,  # Send the file content as the request body.
                timeout=300,
            )
        response.raise_for_status()

    def upload_media_asset(
        self,
        file_path: str,
        kind: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Uploads a media asset to the World Labs API in a two-step process.

        This method orchestrates both `prepare_upload` to get a pre-signed URL
        and `upload_file` to perform the actual file transfer.

        Args:
            file_path: The local path to the media file to upload.
            kind: The type of media asset (e.g., "IMAGE", "3D_MODEL").
            metadata: Optional dictionary of arbitrary metadata to associate with the asset.

        Returns:
            The unique ID of the uploaded media asset.

        Raises:
            FileNotFoundError: If the local file specified by `file_path` does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {file_path}")

        # Extract the file extension, removing the leading dot if present.
        extension = path.suffix.lstrip(".") or None

        # Step 1: Prepare the upload with the API to get upload credentials.
        prepare_response = self.prepare_upload(
            file_name=path.name,
            kind=kind,
            extension=extension,
            metadata=metadata,
        )
        upload_info = prepare_response["upload_info"]

        # Step 2: Upload the file to the provided pre-signed URL.
        self.upload_file(
            file_path=str(path),
            upload_url=upload_info["upload_url"],
            # Provide required headers and method if specified in upload_info, otherwise use defaults.
            required_headers=upload_info.get("required_headers", {}),
            method=upload_info.get("upload_method", "PUT"),
        )
        media_asset = prepare_response["media_asset"]
        # Return the media asset ID, prioritizing 'media_asset_id' but falling back to 'id'.
        return media_asset.get("media_asset_id") or media_asset["id"]

    def generate_world(
        self,
        world_prompt: Dict[str, Any],
        model: str = "marble-1.1",
        display_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initiates the generation of a new 3D world based on a given prompt.

        Args:
            world_prompt: A dictionary defining the prompt for world generation.
                          This typically includes descriptions, asset references, etc.
            model: The generation model to use (e.g., "marble-1.1").
            display_name: An optional display name for the generated world.
            tags: Optional list of tags to categorize the world.
            metadata: Optional dictionary of arbitrary metadata for the world.
            **kwargs: Additional parameters to pass directly to the generation API.

        Returns:
            A dictionary containing the response from the API, typically an `Operation` object.
            This operation can be polled to check the status of the world generation.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "world_prompt": world_prompt,
        }
        # Add optional fields to the payload if they are provided.
        if display_name is not None:
            payload["display_name"] = display_name
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata
        # Include any additional keyword arguments directly in the payload.
        payload.update(kwargs)
        return self._post("/marble/v1/worlds:generate", payload)

    def get_operation(self, operation_id: str) -> Dict[str, Any]:
        """
        Retrieves the status and details of an asynchronous operation.

        Operations are typically returned by long-running API calls like `generate_world`.
        You can poll this endpoint with the `operation_id` to check progress and results.

        Args:
            operation_id: The ID of the operation to retrieve.

        Returns:
            A dictionary containing the `Operation` object, which includes status, progress,
            and eventual results (e.g., a generated world ID).

        Raises:
            requests.exceptions.HTTPError: If the API response status code indicates an error.
        """
        response = self._requests().get(
            self._url(f"/marble/v1/operations/{operation_id}"),
            headers={"WLT-Api-Key": self.api_key},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def get_world(self, world_id: str) -> Dict[str, Any]:
        """
        Retrieves the details of a specific generated world.

        Args:
            world_id: The ID of the world to retrieve.

        Returns:
            A dictionary containing the `World` object, including its properties,
            assets, and generation details.

        Raises:
            requests.exceptions.HTTPError: If the API response status code indicates an error.
        """
        response = self._requests().get(
            self._url(f"/marble/v1/worlds/{world_id}"),
            headers={"WLT-Api-Key": self.api_key},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def download_asset(
        self,
        asset_url: str,
        save_path: str,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Downloads a file from a given URL to a specified local path.

        This method is suitable for downloading large files as it streams the content
        in chunks.

        Args:
            asset_url: The URL of the asset to download.
            save_path: The local file path where the asset should be saved.
            chunk_size: The size of each chunk to read/write during streaming (in bytes).

        Returns:
            The absolute path to the downloaded file as a string.

        Raises:
            requests.exceptions.HTTPError: If the download response status code indicates an error.
        """
        output_path = Path(save_path)
        # Create parent directories if they don't exist, preventing FileNotFoundError.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self._requests().get(asset_url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with output_path.open("wb") as file:
                # Iterate over the response content in chunks and write to the file.
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive new chunks
                        file.write(chunk)
        return str(output_path)

    def save_json(
        self,
        payload: Dict[str, Any],
        save_path: str,
    ) -> str:
        """
        Saves a Python dictionary as a pretty-printed JSON file to a specified local path.

        Args:
            payload: The dictionary to be saved as JSON.
            save_path: The local file path where the JSON should be saved.

        Returns:
            The absolute path to the saved JSON file as a string.
        """
        output_path = Path(save_path)
        # Create parent directories if they don't exist, preventing FileNotFoundError.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            # Dump the dictionary to the file as JSON, ensuring non-ASCII characters are handled
            # and the output is indented for readability.
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return str(output_path)