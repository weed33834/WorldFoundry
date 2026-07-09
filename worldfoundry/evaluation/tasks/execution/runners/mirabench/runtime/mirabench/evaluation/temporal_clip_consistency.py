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


def EvaluateTemporalClipConsistency(clip_model, preprocess, store_image_folder, device):
    cnt = 0
    video_sim = 0.0

    tmp_paths = [os.path.join(store_image_folder, "frames_" + str(f) + ".png") for f in range(1,1+len(os.listdir(store_image_folder)))]
    images = []

    for tmp_path in tmp_paths:
        images.append(preprocess(Image.open(tmp_path)))
    images = torch.stack(images)
    
    with torch.no_grad():
        images = images.to(device)
        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
        for i in range(len(image_features)):
            image_feature = image_features[i].unsqueeze(0)
            if i == 0:
                first_image_feature = image_feature
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
                sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
                cnt += 1
            former_image_feature = image_feature
        sim_per_image = video_sim / cnt
        
    return sim_per_image
