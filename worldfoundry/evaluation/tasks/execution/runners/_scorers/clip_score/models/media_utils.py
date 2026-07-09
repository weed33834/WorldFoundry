import os

import cv2
import numpy as np


def extract_frames(video_path, num_frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // num_frames
    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = video.read()
        if ret:
            output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_path, frame)
        else:
            print(f"Failed to extract frame {i}")
    video.release()


def concatenate_images_horizontal(images, dist_images):
    total_width = sum(img.shape[1] for img in images) + dist_images * (len(images) - 1)
    height = max(img.shape[0] for img in images)
    new_img = np.zeros((height, total_width, 3), dtype=np.uint8)
    current_width = 0
    for img in images:
        h, w = img.shape[:2]
        new_img[:h, current_width : current_width + w] = img
        current_width += w + dist_images
    return new_img
