# import os
# import pickle
# import traceback
# import warnings
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import numpy as np
# import torch
# from decord import VideoReader, cpu
# from torch.utils.data import Dataset
# from torchvision import transforms as T
# from tqdm import tqdm
# import time

# # from dataset_utils import Resize_Preprocess, ToTensorVideo
# from cosmos_predict2.data.dataset_utils import Resize_Preprocess, ToTensorVideo
# import json

# class DatasetPD(Dataset):
#     def __init__(
#         self,
#         dataset_path,
#         t5_dir,
#         num_frames,
#         video_size,
#         start_frame_interval=1,
#     ):
#         """Dataset class for loading image-text-to-video generation data.

#         Args:
#             dataset_dir (str): Base path to the dataset directory
#             sequence_interval (int): Interval between sampled frames in a sequence
#             num_frames (int): Number of frames to load per sequence
#             video_size (list): Target size [H,W] for video frames

#         Returns dict with:
#             - video: RGB frames tensor [T,C,H,W]
#             - video_name: Dict with episode/frame metadata
#         """

#         super().__init__()
#         self.dataset_path = dataset_path
#         self.dataset_dir = os.path.dirname(dataset_path)
#         self.t5_dir = t5_dir
#         self.start_frame_interval = start_frame_interval
#         # self.sequence_interval = sequence_interval
#         self.sequence_length = num_frames

#         start = time.time()
#         f = open(self.dataset_path, "r")
#         all_jsons = [json.loads(line) for line in f]
#         # all_jsons = all_jsons[:1000]
#         self.video_paths = [data["video_path"] for data in all_jsons]
#         print(f"{len(self.video_paths)} videos in total")   

#         self.t5_paths = [os.path.join(self.t5_dir, str(data["index"]) + ".pickle") for data in all_jsons]

#         self.samples = self._init_samples(self.video_paths, self.t5_paths)
#         end = time.time()
#         # self.samples = sorted(self.samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))
#         print(f"{len(self.samples)} samples in total")
#         print(f"Loading Time: {end-start}s")
#         # time.sleep(60*6) # wait for loading, torch launch watchdog timeout
#         self.wrong_number = 0
#         self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

#     def __str__(self):
#         return f"{len(self.video_paths)} samples from {self.dataset_dir}"

#     def _init_samples(self, video_paths, t5_paths):
#         samples = []
#         with ThreadPoolExecutor(64) as executor:
#             future_to_video_path = {
#                 executor.submit(self._load_and_process_video_path, video_path, t5_path): (video_path, t5_path) for video_path, t5_path in zip(video_paths, t5_paths)
#             }
#             for future in tqdm(as_completed(future_to_video_path), total=len(video_paths)):
#                 samples.extend(future.result())
#         return samples

#     def _load_and_process_video_path(self, video_path, t5_path):
#         try:
#             vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)

#             n_frames = len(vr)

#             samples = []
#             if n_frames >= self.sequence_length:
#                 frame_indices = np.linspace(0, n_frames - 1, self.sequence_length)
#                 frame_indices = np.round(frame_indices).astype(int).tolist()
#                 sample = dict()
#                 sample["video_path"] = video_path
#                 sample["t5_embedding_path"] = t5_path
#                 sample["frame_ids"] = frame_indices
#                 samples.append(sample)
#         except:
#             samples = []
#         return samples

#     def __len__(self):
#         return len(self.samples)

#     def _load_video(self, video_path, frame_ids):
#         vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
#         assert (np.array(frame_ids) < len(vr)).all()
#         assert (np.array(frame_ids) >= 0).all()
#         vr.seek(0)
#         frame_data = vr.get_batch(frame_ids).asnumpy()
#         # try:
#         #     fps = vr.get_avg_fps()
#         # except Exception:  # failed to read FPS
#         #     fps = 24
#         fps = 24
#         return frame_data, fps

#     def _get_frames(self, video_path, frame_ids):
#         frames, fps = self._load_video(video_path, frame_ids)
#         frames = frames.astype(np.uint8)
#         frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
#         frames = self.preprocess(frames)
#         frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
#         return frames, fps

#     def __getitem__(self, index):
#         try:
#             sample = self.samples[index]
#             video_path = sample["video_path"]
#             frame_ids = sample["frame_ids"]

#             data = dict()

#             video, fps = self._get_frames(video_path, frame_ids)
#             video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
#             data["video"] = video
#             data["video_name"] = {
#                 "video_path": video_path,
#                 "t5_embedding_path": sample["t5_embedding_path"],
#                 "start_frame_id": str(frame_ids[0]),
#             }

#             # Just add these to fit the interface
#             # t5_embedding = np.load(sample["t5_embedding_path"])[0]
#             with open(sample["t5_embedding_path"], "rb") as f:
#                 t5_embedding = pickle.load(f)[0]  # [n_tokens, 1024]
#             n_tokens = t5_embedding.shape[0]
#             if n_tokens < 512:
#                 t5_embedding = np.concatenate(
#                     [t5_embedding, np.zeros((512 - n_tokens, 1024), dtype=np.float32)], axis=0
#                 )
#             t5_text_mask = torch.zeros(512, dtype=torch.int64)
#             t5_text_mask[:n_tokens] = 1

#             data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)
#             data["t5_text_mask"] = t5_text_mask
#             data["fps"] = fps
#             data["image_size"] = torch.tensor([704, 1280, 704, 1280])
#             data["num_frames"] = self.sequence_length
#             data["padding_mask"] = torch.zeros(1, 704, 1280)

#             return data
#         except Exception:
#             warnings.warn(
#                 f"Invalid data encountered: {self.samples[index]['video_path']}. Skipped "
#                 f"(by randomly sampling another sample in the same dataset)."
#             )
#             warnings.warn("FULL TRACEBACK:")
#             warnings.warn(traceback.format_exc())
#             self.wrong_number += 1
#             print(self.wrong_number)
#             return self[np.random.randint(len(self.samples))]


"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2_wow -> cosmos_predict2 -> data -> dataset_pd.py functionality."""

import os
import pickle
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
import time
from torchvision.transforms.v2 import UniformTemporalSubsample

# from dataset_utils import Resize_Preprocess, ToTensorVideo
from cosmos_predict2.data.dataset_utils import Resize_Preprocess, ToTensorVideo
import json

class DatasetPD(Dataset):
    """Dataset pd implementation."""
    def __init__(
        self,
        dataset_path,
        t5_dir,
        num_frames,
        video_size,
        start_frame_interval=1,
    ):
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_dir = os.path.dirname(dataset_path)
        self.t5_dir = t5_dir
        self.start_frame_interval = start_frame_interval

        self.sequence_length = num_frames

        start = time.time()
        f = open(self.dataset_path, "r")
        all_jsons = [json.loads(line) for line in f]

        self.video_paths = [data["video_path"] for data in all_jsons]
        print(f"{len(self.video_paths)} videos in total")   

        self.t5_paths = [os.path.join(self.t5_dir, str(data["index"]) + ".pickle") for data in all_jsons]

        # self.samples = self._init_samples(self.video_paths, self.t5_paths)
        end = time.time()
        # self.samples = sorted(self.samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))
        # print(f"{len(self.samples)} samples in total")
        print(f"Loading Time: {end-start}s")
        # time.sleep(60*6) # wait for loading, torch launch watchdog timeout
        self.wrong_number = 0
        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

    def __str__(self):
        """Str."""
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self):
        """Len."""
        return len(self.video_paths)

    def _load_video(self, video_path):
        """Helper function to load video.

        Args:
            video_path: The video path.
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        frame_ids = np.linspace(0, len(vr) - 1).astype(np.int32)
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            fps = 24
        return frame_data, fps

    def _get_frames(self, video_path):
        """Helper function to get frames.

        Args:
            video_path: The video path.
        """
        frames, fps = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
        frames = UniformTemporalSubsample(self.sequence_length)(frames)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps


    def __getitem__(self, index):
        """Getitem.

        Args:
            index: The index.
        """
        try:
            video_path = self.video_paths[index]

            data = dict()

            video, fps = self._get_frames(video_path)
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
            data["video"] = video
            
            # t5_embedding_path = os.path.join(
            #     self.t5_dir,
            #     os.path.basename(video_path).replace(".mp4", ".pickle"),
            # )

            t5_embedding_path = os.path.join(
                self.t5_dir,
                f"{(int(os.path.basename(video_path).split('.')[0]) + 1):08d}.pickle",
            )
            data["video_name"] = {
                "video_path": video_path,
                "t5_embedding_path": t5_embedding_path,
            }

            # Just add these to fit the interface
            # t5_embedding = np.load(sample["t5_embedding_path"])[0]

            
            with open(t5_embedding_path, "rb") as f:
                t5_embedding = pickle.load(f)[0]  # [n_tokens, 1024]
            n_tokens = t5_embedding.shape[0]
            if n_tokens < 512:
                t5_embedding = np.concatenate(
                    [t5_embedding, np.zeros((512 - n_tokens, 1024), dtype=np.float32)], axis=0
                )
            t5_text_mask = torch.zeros(512, dtype=torch.int64)
            t5_text_mask[:n_tokens] = 1

            data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)
            data["t5_text_mask"] = t5_text_mask
            data["fps"] = fps
            data["image_size"] = torch.tensor([704, 1280, 704, 1280])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 704, 1280)

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.video_paths[index]}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.video_paths))]
