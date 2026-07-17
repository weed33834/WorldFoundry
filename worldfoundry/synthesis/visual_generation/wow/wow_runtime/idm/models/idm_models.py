import timm
import torch
import torch.nn as nn
import torchvision.models as models

# ------------------------------
# Model definitions
# ------------------------------


class DinoIDM(nn.Module):
    def __init__(self, output_dim=7):
        super().__init__()
        self.encoder = timm.create_model("vit_small_patch14_dinov2", pretrained=True)
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.encoder)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        self.encoder.head = nn.Identity()
        self.embed_dim = self.encoder.num_features

        self.embeds = nn.Embedding(20, 128)
        self.image_fc = nn.Linear(self.embed_dim * 2, 512)
        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, rgb, instr):
        img1 = rgb[:, 0]
        img2 = rgb[:, 1]
        feat1 = self.encoder(self.transforms(img1))
        feat2 = self.encoder(self.transforms(img2))
        x = torch.cat([feat1, feat2], dim=1)
        x = self.image_fc(x)
        instr_embed = self.embeds(instr)
        x = torch.cat([x, instr_embed], dim=1)
        out = self.fc(x)
        return out


# from cotracker.utils.visualizer import Visualizer


class Dino3DFlowIDM(nn.Module):
    def __init__(self, output_dim=7):
        super().__init__()
        # load Co-Tracker model via torch.hub (kept flexible; no hardcoded paths)
        self.flow_tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self.encoder = timm.create_model("vit_small_patch14_dinov2", pretrained=True)

        data_config = timm.data.resolve_model_data_config(self.encoder)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        self.encoder.head = nn.Identity()
        self.embed_dim = self.encoder.num_features

        # Image feature processing
        self.image_fc = nn.Linear(self.embed_dim * 2, 512)

        # Flow feature processing layers
        self.flow_fc = nn.Sequential(nn.Linear(400 * 2, 256), nn.ReLU(), nn.Linear(256, 128))

        # Overall fusion and prediction layers
        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
        )

    # inference forward: accepts raw RGB pair tensors and runs flow tracker internally
    def infer_forward(self, rgb, instr=None):
        B = rgb.size(0)
        img1 = rgb[:, 0]  # B C H W
        img2 = rgb[:, 1]  # B C H W

        # Image feature extraction (batched)
        feat1 = self.encoder(self.transforms(img1))  # B, D
        feat2 = self.encoder(self.transforms(img2))  # B, D
        img_feat = torch.cat([feat1, feat2], dim=1)  # B, 2D
        img_feat = self.image_fc(img_feat)  # B, 512

        # Flow feature extraction (processed per-sample to avoid CoTracker batching issues)
        flow_feats = []
        for i in range(B):
            rgb_i = rgb[i : i + 1]  # shape: (1, 2, C, H, W)
            rgb_i_repeat = rgb_i.repeat(1, 5, 1, 1, 1) * 255

            # Single inference through flow tracker
            pred_tracks, pred_visibility = self.flow_tracker(rgb_i_repeat, grid_size=20)  # (1, 2, 400, 2)

            # Use the last frame's track points as the flow feature
            tracks = pred_tracks[0, -1]  # shape: (400, 2)
            flow_feat = tracks.reshape(-1)  # shape: (400*2,)
            flow_feats.append(flow_feat)

        # Stack and reduce flow features
        flow_feats = torch.stack(flow_feats, dim=0)  # shape: (B, 800)
        flow_feats = self.flow_fc(flow_feats)  # shape: (B, 128)

        # Concatenate image and flow features and predict
        x = torch.cat([img_feat, flow_feats], dim=1)  # B, 640
        out = self.fc(x)
        return out


class ResNetIDM(nn.Module):
    def __init__(self, input_dim=3, output_dim=7):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

        self.model = nn.Sequential(*list(self.model.children()))[:7]
        self.embeds = nn.Embedding(20, 128)
        self.down = nn.Conv2d(256, 256, 3, 2, 1)

        self.image_fc = nn.Linear(8192, 512)
        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, rgb, instr):
        # Input shape: B T C H W
        first = self.down(self.model(rgb[:, 0]))
        first = torch.flatten(first, 1)
        second = self.down(self.model(rgb[:, 1]))
        second = torch.flatten(second, 1)
        instr = self.embeds(instr)
        x = torch.cat([first, second], 1)
        x = self.image_fc(x)
        x = torch.cat([x, instr], 1)
        x = self.fc(x)
        return x


def cycle(dl):
    while True:
        for data in dl:
            yield data
