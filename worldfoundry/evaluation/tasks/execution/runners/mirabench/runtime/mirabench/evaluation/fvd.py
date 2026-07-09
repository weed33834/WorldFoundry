import os
import numpy as np
import argparse
from tqdm import tqdm
import copy
import json
import cv2
import glob
from PIL import Image
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import sys

from einops import rearrange
from sklearn.metrics.pairwise import polynomial_kernel

from evaluation.pytorch_i3d import InceptionI3d

MAX_BATCH = 8 #16
FVD_SAMPLE_SIZE = 2048
TARGET_RESOLUTION = (224, 224)

def preprocess(videos, target_resolution):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, h, w, c = videos.shape
    all_frames = torch.FloatTensor(videos).flatten(end_dim=1) # (b * t, h, w, c)
    all_frames = all_frames.permute(0, 3, 1, 2).contiguous() # (b * t, c, h, w)
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    resized_videos = resized_videos.view(b, t, c, *target_resolution)
    output_videos = resized_videos.transpose(1, 2).contiguous() # (b, c, t, *)
    scaled_videos = 2. * output_videos / 255. - 1 # [-1, 1]
    return scaled_videos

def preprocess2(videos, target_resolution):
    # videos in tensor in -1~1
    all_frames = rearrange(videos, 'b c t h w -> (b t) c h w')
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    return resized_videos

def preprocess_styleganv_i3d(videos, target_resolution):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, h, w, c = videos.shape
    all_frames = torch.FloatTensor(videos).flatten(end_dim=1) # (b * t, h, w, c)
    all_frames = all_frames.permute(0, 3, 1, 2).contiguous() # (b * t, c, h, w)
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    resized_videos = resized_videos.view(b, t, c, *target_resolution)
    output_videos = resized_videos.transpose(1, 2).contiguous() # (b, c, t, *)
    # scaled_videos = 2. * output_videos / 255. - 1 # [-1, 1]
    return output_videos

def get_fvd_logits(videos, i3d, device, batch_size=None):
    videos = preprocess(videos, TARGET_RESOLUTION)
    # videos = preprocess_styleganv_i3d(videos, TARGET_RESOLUTION)
    embeddings = get_logits(i3d, videos, device, batch_size=batch_size)
    return embeddings

def load_fvd_model(device, i3d_path):
    i3d = InceptionI3d(400, in_channels=3).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if i3d_path is None:
        i3d_path = os.path.join(current_dir, 'i3d_pretrained_400.pt')
        assert(os.path.exists(i3d_path))
    i3d.load_state_dict(torch.load(i3d_path, map_location=device))
    i3d.eval()
    return i3d

def load_stylegan_v_i3d(device):
    fpath = os.environ.get("WORLDFOUNDRY_MIRABENCH_I3D_TORCHSCRIPT")
    if not fpath:
        fpath = str(Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", "~/.cache/huggingface/hub")).expanduser() / "i3d_torchscript.pt")
    return torch.jit.load(fpath).eval().to(device)

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1) # unbiased estimate
    m_center = m - torch.mean(m, dim=1, keepdim=True)
    mt = m_center.t()  # if complex: mt = m.t().conj()
    return fact * m_center.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = torch.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd


def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)
    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)
    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum
    return mmd


def get_logits(i3d, videos, device, batch_size=None):
    detector_kwargs = dict(rescale=True, resize=True, return_features=True)# 3
    if batch_size is None:
        batch_size = MAX_BATCH
    # assert videos.shape[0] % batch_size == 0, f'{videos.shape[0]}, {batch_size}'
    with torch.no_grad():
        logits = []
        for i in range(0, videos.shape[0], batch_size):
            batch = videos[i:i + batch_size].to(device)
            logits.append(i3d(batch))
            # logits.append(i3d(batch, **detector_kwargs))#4
        logits = torch.cat(logits, dim=0)
        return logits


def compute_fvd(real, samples, i3d, device=torch.device('cpu')):
    # real, samples are (N, T, H, W, C) numpy arrays in np.uint8
    real, samples = preprocess(real, (224, 224)), preprocess(samples, (224, 224))
    first_embed = get_logits(i3d, real, device)
    second_embed = get_logits(i3d, samples, device)

    return frechet_distance(first_embed, second_embed)


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.target_resolution = (224, 224)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i,select_frame=100):
        frame_path = self.files[i]
        frames_list = ["frames_" + str(f) + ".png" for f in range(1,1+len(os.listdir(frame_path)))]
        if len(frames_list)>select_frame:
            frames_list = [frames_list[i] for i in np.linspace(0,len(frames_list)-1,select_frame).astype(int)]
        video = []
        for frame in frames_list:
            img = Image.open(os.path.join(frame_path, frame)).convert("RGB")
            img = np.array(img)
            img = torch.FloatTensor(img).permute(2, 0, 1)
            img = F.interpolate(img[None], size=self.target_resolution,
                                    mode='bilinear', align_corners=False)
            video.append(img)
        video = torch.cat(video, dim=0).permute(1, 0, 2, 3)
        video = (video/255) * 2. - 1.
        return video


# def load_frame_path_from_dir(datadir):
#     files = glob.glob(os.path.join(datadir, "*", "frames"))
#     files.sort()
#     return files

def get_logits(i3d, frame_dir, device, batch_size, num_workers):
    frame_list=[os.path.join(frame_dir,video_path) for video_path in os.listdir(frame_dir)]

    dataset = VideoDataset(frame_list)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    # if len(dataset) % batch_size != 0:
    #     print("Ignore last")

    with torch.no_grad():
        logits = []
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            logits.append(i3d(batch))
            # logits.append(i3d(batch, **detector_kwargs))#4
        logits = torch.cat(logits, dim=0)
        return logits



def _resolve_fvd_i3d_model_path(ckpt_path):
    fvd_i3d_model_path=os.path.join(ckpt_path,"fvd/i3d_pretrained_400.pt")
    candidates = [
        os.environ.get("WORLDFOUNDRY_MIRABENCH_FVD_I3D_CKPT"),
        fvd_i3d_model_path,
        str(Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", "~/.cache/huggingface/hub")).expanduser() / "MiraBench" / "fvd" / "i3d_pretrained_400.pt"),
        str(Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", "~/.cache/huggingface/hub")).expanduser() / "i3d_pretrained_400.pt"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates if path)
    raise FileNotFoundError(
        "MiraBench FVD requires i3d_pretrained_400.pt to be staged before evaluation.\n"
        f"Searched:\n{searched}\n"
        "Set WORLDFOUNDRY_MIRABENCH_FVD_I3D_CKPT to the checkpoint path if it is stored elsewhere."
    )


def EvaluateFVD(store_image_folder, store_gt_image_folder, ckpt_path, device):
    fvd_i3d_model_path = _resolve_fvd_i3d_model_path(ckpt_path)
    i3d = load_fvd_model(device, fvd_i3d_model_path)

    res_embed = get_logits(i3d, store_image_folder, device, 1, 1)
    gt_embed = get_logits(i3d, store_gt_image_folder, device, 1, 1)

    return frechet_distance(res_embed, gt_embed).item(),polynomial_mmd(res_embed.cpu(), gt_embed.cpu()) # fvd kvd
