import torch
import numpy as np
import json
from PIL import Image, PngImagePlugin
import einops
import cv2
import os
from basicsr.utils import img2tensor
from datasets import load_from_disk
import random

LARGE_ENOUGH_NUMBER = 1024
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)


class DepthDataset():
    def __init__(self, root, split, image_size, p_drop=0.5):
        super(DepthDataset, self).__init__()

        self.root = root
        self.split = split
        self.base_dataset = load_from_disk(self.root)[self.split]
        self.image_size = image_size
        self.p_drop = p_drop

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]
        pil_image = row["image"].convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        image = np.array(pil_image)
        pil_depth_dimage = row["control_depth"].resize((self.image_size, self.image_size), Image.Resampling.NEAREST).convert("RGB")
        depth = np.array(pil_depth_dimage)
        sentence = row["text"]
        if random.random() < self.p_drop:
            sentence = ""

        image = einops.rearrange(torch.from_numpy(image.astype(np.float32) / 255.), "h w c ->  c h w")
        depth = einops.rearrange(torch.from_numpy(depth.astype(np.float32) / 255.), "h w c ->  c h w")

        return {'im': image, 'depth': depth, 'sentence': sentence}
