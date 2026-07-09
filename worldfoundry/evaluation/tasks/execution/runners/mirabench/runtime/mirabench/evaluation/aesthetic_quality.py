import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from worldfoundry.base_models.capabilities import vbench_asset_path
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

def get_aesthetic_model(cache_folder):
    """load the aethetic model"""
    path_to_model = cache_folder + "/aesthetic_model/sa_0_4_vit_l_14_linear.pth"
    if not os.path.exists(path_to_model):
        path_to_model = str(vbench_asset_path("vbench_aesthetic_linear_checkpoint"))
    m = nn.Linear(768, 1)
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def EvaluateLaionAesthetic(aesthetic_model, clip_model, preprocess, store_image_folder, device):
    aesthetic_model.eval()
    aesthetic_model=aesthetic_model.to(clip_model.dtype)
    clip_model.eval()
    with torch.no_grad():
        tmp_paths = [os.path.join(store_image_folder, "frames_" + str(f) + ".png") for f in range(1,1+len(os.listdir(store_image_folder)))]
        images = []

        for tmp_path in tmp_paths:
            images.append(preprocess(Image.open(tmp_path)))
        images = torch.stack(images)
            
        images = images.to(device)
        scores=[]
        for i in range(images.shape[0]):
            image_features = clip_model.encode_image(images[[i]])
            image_features = F.normalize(image_features, dim=-1, p=2)
            aesthetic_scores = aesthetic_model(image_features).squeeze()
            scores.append(aesthetic_scores.item())
    return np.mean(scores)
