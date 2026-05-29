import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
from PIL import Image, PngImagePlugin
import einops
import cv2
import os

from annotator.canny import CannyDetector
from basicsr.utils import img2tensor
from datasets import load_from_disk
import random

LARGE_ENOUGH_NUMBER = 1024
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)


class CannyControl(Dataset):
    
    def __init__(self, metadata_path, image_size, p_drop=0.5):
        super().__init__()
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        self.ade20k_palette = np.load("ade20k_palette.npy")
        self.canny_detector = CannyDetector()
        self.image_size = image_size
        self.p_drop = p_drop

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        pil_image = Image.open(row.path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        image = np.array(pil_image)

        pil_seg_image = Image.open(row.seg_path).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        seg_image = self.ade20k_palette[np.array(pil_seg_image)].astype(np.uint8)
        canny = np.array(Image.fromarray(self.canny_detector(seg_image, 100, 200)).convert("RGB"))
        sentence = row.caption
        if random.random() < self.p_drop:
            sentence = ""

        image = einops.rearrange(torch.from_numpy(image.astype(np.float32) / 255.), "h w c ->  c h w")
        canny = einops.rearrange(torch.from_numpy(canny.astype(np.float32) / 255.), "h w c ->  c h w")

        return {'im': image, 'canny': canny, 'sentence': sentence}
