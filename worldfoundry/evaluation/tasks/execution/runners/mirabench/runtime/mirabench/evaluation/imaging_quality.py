import torch
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ

def transform(images, preprocess_mode='shorter'):
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h,w) > 512:
            scale = 512./min(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)
            if preprocess_mode == 'shorter_centercrop':
                images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h,w) > 512:
            scale = 512./max(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)

    elif preprocess_mode == 'None':
        return images / 255.

    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.


def EvaluateImagingQuality(imaging_quality_model,store_image_folder,device):
    tmp_paths = [os.path.join(store_image_folder, "frames_" + str(f) + ".png") for f in range(1,1+len(os.listdir(store_image_folder)))]
    images = []

    for tmp_path in tmp_paths:
        images.append(np.array(Image.open(tmp_path).convert('RGB')).astype(np.uint8))

    images=np.array(images)
    images = torch.Tensor(images)
    images = images.permute(0, 3, 1, 2)

    images = transform(images, "longer")
    acc_score_video = 0.
    for i in range(len(images)):
        with torch.no_grad():
            frame = images[i].unsqueeze(0).to(device)
            score = imaging_quality_model(frame)
            acc_score_video += float(score)

    return acc_score_video/len(images)
    
