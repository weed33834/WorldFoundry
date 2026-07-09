import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.utilities.data import dim_zero_cat

from PIL import Image
import torchvision.transforms as TF
from tqdm import tqdm
import json
import os



def load_frame_path_from_dir(datadir,select_frame=100):
    dir_list = [os.path.join(datadir,video_path) for video_path in os.listdir(datadir)]
    all_files=[]
    for dir in dir_list:
        files=[os.path.join(dir, f) for f in os.listdir(dir)]
        files.sort()
        if len(files)>select_frame:
            files=[files[i] for i in np.linspace(0,len(files)-1,select_frame).astype(int)]
        all_files+=files

    return all_files


def EvaluateFID(store_image_folder, store_gt_image_folder, ckpt_path, device):

    fid_image_transforms=TF.Compose([
                TF.Resize((299,299)),
                # TF.CenterCrop(299),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


    store_image_folder_files = load_frame_path_from_dir(store_image_folder)
    image_dataset=[]
    for image_path_i in store_image_folder_files:
        img = Image.open(image_path_i).convert('RGB')
        image_dataset.append(fid_image_transforms(img).unsqueeze(0).to(device))
    # image_dataset=torch.concat(image_dataset).to(device)

    store_gt_image_folder_files = load_frame_path_from_dir(store_gt_image_folder)
    gt_image_dataset=[]
    for image_path_i in store_gt_image_folder_files:
        img = Image.open(image_path_i).convert('RGB')
        gt_image_dataset.append(fid_image_transforms(img).unsqueeze(0).to(device))
    # gt_image_dataset=torch.concat(gt_image_dataset).to(device)
    

    fid_model=FrechetInceptionDistance(normalize=True).to(device)
    with torch.no_grad():
        for gt_image_tensor in gt_image_dataset:
            fid_model.update(gt_image_tensor,real=True)
        for image_tensor in image_dataset:
            fid_model.update(image_tensor,real=False)

    fid_score=fid_model.compute()

    kid_model = KernelInceptionDistance(normalize=True, subset_size=100).to(device)
    with torch.no_grad():
        for gt_image_tensor in gt_image_dataset:
            kid_model.update(gt_image_tensor, real=True)
        for image_tensor in image_dataset:
            kid_model.update(image_tensor, real=False)
    if dim_zero_cat(kid_model.fake_features).shape[0]<kid_model.subset_size:
        raise Exception("kid subset size is too big!")

    kid_score=kid_model.compute()

    return fid_score.item(), kid_score[0].item()
