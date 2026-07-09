"""Module for base_models -> diffusion_model -> diffsynth -> extensions -> FastBlend -> data.py functionality."""

import imageio, os
import numpy as np
from PIL import Image


def read_video(file_name):
    """Read video.

    Args:
        file_name: The file name.
    """
    reader = imageio.get_reader(file_name)
    video = []
    for frame in reader:
        frame = np.array(frame)
        video.append(frame)
    reader.close()
    return video


def get_video_fps(file_name):
    """Get video fps.

    Args:
        file_name: The file name.
    """
    reader = imageio.get_reader(file_name)
    fps = reader.get_meta_data()["fps"]
    reader.close()
    return fps


def save_video(frames_path, video_path, num_frames, fps):
    """Save video.

    Args:
        frames_path: The frames path.
        video_path: The video path.
        num_frames: The num frames.
        fps: The fps.
    """
    writer = imageio.get_writer(video_path, fps=fps, quality=9)
    for i in range(num_frames):
        frame = np.array(Image.open(os.path.join(frames_path, "%05d.png" % i)))
        writer.append_data(frame)
    writer.close()
    return video_path


class LowMemoryVideo:
    """Low memory video implementation."""
    def __init__(self, file_name):
        """Init.

        Args:
            file_name: The file name.
        """
        self.reader = imageio.get_reader(file_name)
    
    def __len__(self):
        """Len."""
        return self.reader.count_frames()

    def __getitem__(self, item):
        """Getitem.

        Args:
            item: The item.
        """
        return np.array(self.reader.get_data(item))

    def __del__(self):
        """Del."""
        self.reader.close()


def split_file_name(file_name):
    """Split file name.

    Args:
        file_name: The file name.
    """
    result = []
    number = -1
    for i in file_name:
        if ord(i)>=ord("0") and ord(i)<=ord("9"):
            if number == -1:
                number = 0
            number = number*10 + ord(i) - ord("0")
        else:
            if number != -1:
                result.append(number)
                number = -1
            result.append(i)
    if number != -1:
        result.append(number)
    result = tuple(result)
    return result


def search_for_images(folder):
    """Search for images.

    Args:
        folder: The folder.
    """
    file_list = [i for i in os.listdir(folder) if i.endswith(".jpg") or i.endswith(".png")]
    file_list = [(split_file_name(file_name), file_name) for file_name in file_list]
    file_list = [i[1] for i in sorted(file_list)]
    file_list = [os.path.join(folder, i) for i in file_list]
    return file_list


def read_images(folder):
    """Read images.

    Args:
        folder: The folder.
    """
    file_list = search_for_images(folder)
    frames = [np.array(Image.open(i)) for i in file_list]
    return frames


class LowMemoryImageFolder:
    """Low memory image folder implementation."""
    def __init__(self, folder, file_list=None):
        """Init.

        Args:
            folder: The folder.
            file_list: The file list.
        """
        if file_list is None:
            self.file_list = search_for_images(folder)
        else:
            self.file_list = [os.path.join(folder, file_name) for file_name in file_list]
    
    def __len__(self):
        """Len."""
        return len(self.file_list)

    def __getitem__(self, item):
        """Getitem.

        Args:
            item: The item.
        """
        return np.array(Image.open(self.file_list[item]))

    def __del__(self):
        """Del."""
        pass


class VideoData:
    """Video data implementation."""
    def __init__(self, video_file, image_folder, **kwargs):
        """Init.

        Args:
            video_file: The video file.
            image_folder: The image folder.
        """
        if video_file is not None:
            self.data_type = "video"
            self.data = LowMemoryVideo(video_file, **kwargs)
        elif image_folder is not None:
            self.data_type = "images"
            self.data = LowMemoryImageFolder(image_folder, **kwargs)
        else:
            raise ValueError("Cannot open video or image folder")
        self.length = None
        self.height = None
        self.width = None

    def raw_data(self):
        """Raw data."""
        frames = []
        for i in range(self.__len__()):
            frames.append(self.__getitem__(i))
        return frames

    def set_length(self, length):
        """Set length.

        Args:
            length: The length.
        """
        self.length = length

    def set_shape(self, height, width):
        """Set shape.

        Args:
            height: The height.
            width: The width.
        """
        self.height = height
        self.width = width

    def __len__(self):
        """Len."""
        if self.length is None:
            return len(self.data)
        else:
            return self.length

    def shape(self):
        """Shape."""
        if self.height is not None and self.width is not None:
            return self.height, self.width
        else:
            height, width, _ = self.__getitem__(0).shape
            return height, width

    def __getitem__(self, item):
        """Getitem.

        Args:
            item: The item.
        """
        frame = self.data.__getitem__(item)
        height, width, _ = frame.shape
        if self.height is not None and self.width is not None:
            if self.height != height or self.width != width:
                frame = Image.fromarray(frame).resize((self.width, self.height))
                frame = np.array(frame)
        return frame

    def __del__(self):
        """Del."""
        pass
