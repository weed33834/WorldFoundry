"""
Module for interacting with the MiniMax video generation API.

This module provides a client class, `Minimax`, to submit video generation tasks,
query their status, and fetch the results. It handles API key authentication
and structured logging of API events.
"""

import os
import time
import json
import base64
from typing import Literal

from worldfoundry.core.io import extract_frames_from_video_url


def _log_api_event(event: str, payload: dict[str, object]) -> None:
    """
    Print a small structured API status event.

    Args:
        event: Stable event name for the MiniMax runtime step.
        payload: Non-sensitive fields safe to display in CLI logs.
    """
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True))


class Minimax:
    """
    A client for interacting with the MiniMax video generation API.

    This class provides methods to initialize the API client, invoke video
    generation tasks, query the status of ongoing tasks, fetch the final
    video results, and orchestrate the full generation process.
    """

    def __init__(
        self,
        model_name: str,
        generation_type: Literal["t2v", "i2v", "v2v"],
        url: str,
        model: str,
    ):
        """
        Initializes the MiniMax API client.

        Args:
            model_name: The name of the MiniMax model to use (e.g., 'MiniMax-Video').
            generation_type: The type of video generation (text-to-video 't2v',
                             image-to-video 'i2v', or video-to-video 'v2v').
            url: The base URL for the MiniMax video generation API endpoint.
            model: The specific model identifier for the generation task.
        """
        self.api_key = os.environ["MINIMAX_API_KEY"]
        self.model_name = model_name
        self.generation_type = generation_type
        self.url = url
        self.model = model
        
    def invoke_video_generation(self, prompt: str, image_path: str)->str:
        """
        Submits a video generation task to the MiniMax API.

        This method sends the prompt and an initial image to the API to start
        the video generation process and returns a task ID.

        Args:
            prompt: The text prompt describing the desired video content.
            image_path: The file path to the initial image for I2V generation.

        Returns:
            str: The task ID returned by the API, used to query the status.
        """
        import requests

        print("-----------------Submit video generation task-----------------")

        # Read the image file, base64 encode it, and decode to UTF-8 for JSON embedding.
        with open(image_path, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Construct the payload for the API request, embedding the base64 image data.
        payload = json.dumps({
            "prompt": prompt,
            "model": self.model,
            "first_frame_image":f"data:image/jpeg;base64,{data}"
        })
        
        # Set up authorization header with the API key.
        headers = {
            'authorization': 'Bearer ' + self.api_key,
            'content-type': 'application/json',
        }

        # Make the POST request to the video generation endpoint.
        response = requests.request("POST", self.url, headers=headers, data=payload)
        response_payload = response.json()
        task_id = response_payload['task_id']
        
        # Log the submission event for tracking.
        _log_api_event(
            "minimax.video_generation.submitted",
            {"status_code": response.status_code, "has_task_id": True},
        )
        return task_id

    def query_video_generation(self, task_id: str):
        """
        Queries the status of an ongoing video generation task.

        Args:
            task_id: The ID of the task to query, obtained from `invoke_video_generation`.

        Returns:
            tuple[str, str]: A tuple containing:
                - The `file_id` (if status is 'Success'), otherwise an empty string.
                - The current status of the task (e.g., 'Preparing', 'Processing', 'Finished', 'Fail').
        """
        import requests

        # Construct the query URL with the given task ID.
        url = "https://api.minimaxi.chat/v1/query/video_generation?task_id="+task_id
        headers = {
            'authorization': 'Bearer ' + self.api_key
        }
        
        # Make the GET request to query the task status.
        response = requests.request("GET", url, headers=headers)
        response_payload = response.json()
        status = response_payload['status']
        
        # Handle different status responses.
        if status == 'Preparing':
            print("...Preparing...")
            return "", 'Preparing'   
        elif status == 'Queueing':
            print("...In the queue...")
            return "", 'Queueing'
        elif status == 'Processing':
            print("...Generating...")
            return "", 'Processing'
        elif status == 'Success':
            # If successful, return the file_id and 'Finished' status.
            return response_payload['file_id'], "Finished"
        elif status == 'Fail':
            return "", "Fail"
        else:
            return "", "Unknown"
    
    def fetch_video_result(self, file_id: str):
        """
        Fetches the generated video result using its file ID.

        This method retrieves the download URL for the video and then extracts
        frames from it.

        Args:
            file_id: The ID of the generated video file, obtained from `query_video_generation`
                     when the status is 'Success'.

        Returns:
            list[PIL.Image.Image]: A list of PIL Image objects representing the frames
                                    extracted from the downloaded video.
        """
        import requests

        print("---------------Video generated successfully, downloading now---------------")
        # Construct the URL to retrieve the file information.
        url = "https://api.minimaxi.chat/v1/files/retrieve?file_id="+file_id
        headers = {
            'authorization': 'Bearer '+ self.api_key,
        }

        # Make the GET request to get the file's metadata, including its download URL.
        response = requests.request("GET", url, headers=headers)

        response_payload = response.json()
        download_url = response_payload['file']['download_url']
        
        # Log the readiness of the video result.
        _log_api_event("minimax.video_result.ready", {"status_code": response.status_code, "has_download_url": True})
        
        # Extract frames from the video URL using a utility function.
        frames = extract_frames_from_video_url(download_url)
        return frames
    
    def generate_video(
        self, 
        prompt: str, 
        image_path: str
    ):
        """
        Orchestrates the entire video generation process from submission to result retrieval.

        This method combines `invoke_video_generation`, `query_video_generation`,
        and `fetch_video_result` into a single, blocking call that polls the API
        until the video is generated or fails.

        Args:
            prompt: The text prompt for the video generation.
            image_path: The path to the initial image for I2V generation.

        Returns:
            list[PIL.Image.Image]: A list of PIL Image objects representing the frames
                                    of the generated video, or an empty list if generation fails.
        """
        # Submit the video generation task and get the task ID.
        task_id = self.invoke_video_generation(prompt, image_path)
        print("-----------------Video generation task submitted -----------------")
        
        frames = []
        # Polling loop to check the status of the video generation task.
        while True:
            time.sleep(10) # Wait for 10 seconds before polling again.

            file_id, status = self.query_video_generation(task_id)
            if file_id != "":
                # If a file_id is returned, generation was successful; fetch the video frames.
                frames = self.fetch_video_result(file_id)
                print("---------------Successful---------------")
                break
            elif status == "Fail" or status == "Unknown":
                # If the status indicates failure or an unknown state, break the loop.
                print("---------------Failed---------------")
                break
            
        return frames