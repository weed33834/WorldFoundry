"""
This module provides an interface for interacting with RunwayML's Gen3 models
to generate videos from text prompts and/or images.
It encapsulates the logic for model initialization, task submission,
and polling for results from the RunwayML API.
"""
import time
import base64
from typing import Literal

from worldfoundry.core.io import extract_frames_from_video_url


class Gen3:
    """
    A client class for interacting with RunwayML's Gen3 video generation models.

    This class handles initializing the RunwayML client and provides a method
    to submit video generation tasks and retrieve the results as video frames.
    """
    def __init__(
        self,
        model_name: str,
        generation_type: Literal["t2v", "i2v", "v2v"],
        model_id: str,
    ):
        """
        Initializes the Gen3 client with specific model details and connects to RunwayML.

        Args:
            model_name (str): The human-readable name of the Gen3 model (e.g., "Gen3 Alpha Turbo").
            generation_type (Literal["t2v", "i2v", "v2v"]): The type of video generation,
                                                            text-to-video (t2v), image-to-video (i2v),
                                                            or video-to-video (v2v).
            model_id (str): The unique identifier for the Gen3 model used by the RunwayML API.
        """
        self.model_name = model_name
        self.generation_type = generation_type
        self.model_id = model_id
        # Lazily import RunwayML client to avoid unnecessary imports if Gen3 is not used,
        # and to ensure it's imported within the scope where it's needed.
        from runwayml import RunwayML

        self.client = RunwayML()
        
    def generate_video(
        self,
        prompt: str,
        image_path: str | None,
    ):    
        """
        Generates a video using the configured RunwayML Gen3 model based on a text prompt and an optional image.

        This method submits an image-to-video task to the RunwayML API,
        polls for its completion, and extracts frames from the resulting video.

        Args:
            prompt (str): The text prompt describing the desired video content.
            image_path (str | None): The file path to an input image (e.g., PNG).
                                     This is typically required for 'i2v' or 'v2v' generation types.

        Returns:
            list: A list of processed frames (e.g., numpy arrays) extracted from the generated video.
        """
        # Read the image file in binary mode and encode it to a base64 string.
        # This format is required for embedding images directly in API requests.
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Create a new image-to-video generation task with the RunwayML API.
        task = self.client.image_to_video.create(
        model=self.model_id,
        # Construct the data URI for the base64 encoded image.
        prompt_image=f"data:image/png;base64,{base64_image}",
        prompt_text=prompt,
        )
        task_id = task.id

        # Wait for an initial period to allow the task to start processing on the server.
        time.sleep(10)
        task = self.client.tasks.retrieve(task_id)
        # Continuously poll the task status until it either succeeds or fails.
        # This loop prevents proceeding until the video generation is complete.
        while task.status not in ['SUCCEEDED', 'FAILED']:
            # Pause for a duration before the next poll to avoid overwhelming the API
            # and to conserve resources.
            time.sleep(10)
            task = self.client.tasks.retrieve(task_id)

        print('Task complete:', task)
        video_url = task.output[0]
        # Extract individual frames from the URL of the newly generated video.
        frames = extract_frames_from_video_url(video_url)

        return frames