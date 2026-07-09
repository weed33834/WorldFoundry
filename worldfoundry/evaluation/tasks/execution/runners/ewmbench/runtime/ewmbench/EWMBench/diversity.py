import clip
import torch
import torch.nn.functional as F
from EWMBench.utils import clip_transform_nocrop, load_dimension_info, load_video
from itertools import combinations
from tqdm import tqdm

from .distributed import get_rank


def compute_cost_matrix(results):
    cost_matrices = {}

    for task_id, episodes in results.items():
        for episode_id, gids in episodes.items():
            gid_list = sorted(gids.keys(), key=int)
            gid_pairs = list(combinations(gid_list, 2))
            gid_cost_list = []

            for gid_a, gid_b in gid_pairs:
                features_a = gids[gid_a].flatten()
                features_b = gids[gid_b].flatten()

                cos_sim = F.cosine_similarity(features_a.unsqueeze(0), features_b.unsqueeze(0)).item()
                cost = 1 - cos_sim
                gid_cost_list.append(cost)

            avg_cost = sum(gid_cost_list) / len(gid_cost_list) if gid_cost_list else 0.0

            if task_id not in cost_matrices:
                cost_matrices[task_id] = {}
            cost_matrices[task_id][episode_id] = avg_cost

    return cost_matrices


def diversity(clip_model, preprocess, video_list, device):
    results = {}
    image_transform = clip_transform_nocrop(224)

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        parts = video_path.split("/")
        task_id = parts[-4]
        episode_id = parts[-3]
        gid = parts[-2]

        if task_id not in results:
            results[task_id] = {}
        if episode_id not in results[task_id]:
            results[task_id][episode_id] = {}

        images = load_video(video_path)

        padded_images = []
        for image in images:
            max_h, max_w = image.shape[1], image.shape[2]
            max_dim = max(max_h, max_w)
            pad_w_left = (max_dim - max_w) // 2
            pad_w_right = max_dim - max_w - pad_w_left
            pad_h_top = (max_dim - max_h) // 2
            pad_h_bottom = max_dim - max_h - pad_h_top
            padded = F.pad(image, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), "constant", 0)
            padded_images.append(padded)

        images = torch.stack(padded_images)
        images = image_transform(images).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1, p=2)
            video_feature = image_features.mean(dim=0).detach().cpu()

        results[task_id][episode_id][gid] = video_feature

        del images, image_features, video_feature
        torch.cuda.empty_cache()

    return compute_cost_matrix(results)


def compute_diversity(json_dir, submodules_list, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_path = submodules_list["model"]

    clip_model, preprocess = clip.load(checkpoint_path, device=device)

    print("Loaded CLIP model for diversity evaluation")

    video_list = load_dimension_info(json_dir, dimension="diversity")

    return diversity(clip_model, preprocess, video_list, device)
