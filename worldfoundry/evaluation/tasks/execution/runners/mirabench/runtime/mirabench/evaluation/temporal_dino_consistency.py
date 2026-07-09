import io
import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage

def dino_transform_Image(n_px):
    return Compose([
        Resize(size=n_px),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def EvaluateTemporalDinoConsistency(dino_model, store_image_folder, device):
    cnt = 0
    video_sim = 0.0
    image_transform = dino_transform_Image(224)

    tmp_paths = [os.path.join(store_image_folder, "frames_" + str(f) + ".png") for f in range(1,1+len(os.listdir(store_image_folder)))]
    images = []
    for tmp_path in tmp_paths:
        images.append(image_transform(Image.open(tmp_path)))

    for i in range(len(images)):
        with torch.no_grad():
            image = images[i].unsqueeze(0)
            image = image.to(device)
            image_features = dino_model(image)
            image_features = F.normalize(image_features, dim=-1, p=2)
            if i == 0:
                first_image_features = image_features
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
                cnt += 1
        former_image_features = image_features

    sim_per_frame = video_sim / cnt
    return sim_per_frame
