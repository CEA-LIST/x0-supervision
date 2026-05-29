import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
import numpy as np
import pandas as pd
import skimage
import cv2
from PIL import Image, PngImagePlugin
from annotator.midas import MidasDetector
from annotator.canny import CannyDetector
import json
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from typing import Union, Tuple, Optional
import os
import random

LARGE_ENOUGH_NUMBER = 1024
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

## Utils ##
def center_crop_and_resize(image, image_size, interpolation):
    # Get original image dimensions
    original_width, original_height = image.size

    # Calculate the aspect ratios
    original_ratio = original_width / original_height
    target_ratio = image_size[1] / image_size[0]

    # Determine the crop box coordinates
    if original_ratio > target_ratio:
        # Original image is wider than the target aspect ratio.
        # We will crop from the sides.
        crop_height = original_height
        crop_width = int(original_height * target_ratio)
        crop_x = (original_width - crop_width) // 2
        crop_y = 0
    else:
        # Original image is taller or equal in ratio to the target.
        # We will crop from the top and bottom.
        crop_width = original_width
        crop_height = int(original_width / target_ratio)
        crop_x = 0
        crop_y = (original_height - crop_height) // 2

    # Perform the crop
    cropped_img = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    
    # Resize the cropped image to the final dimensions
    resized_img = cropped_img.resize((image_size[1], image_size[0]), interpolation)

    return resized_img

    
class DepthControl(Dataset):
    
    def __init__(self, root, split, image_size, p_drop=0.5):
        super().__init__()
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
        # pil_image = center_crop_and_resize(pil_image, (self.image_size, self.image_size), Image.Resampling.BICUBIC)
        image = np.array(pil_image)
        # depth, _  = self.midas(image)
        pil_depth_dimage = row["control_depth"].resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        depth = np.array(pil_depth_dimage)
        prompt = row["text"]
        if random.random() < self.p_drop:
            prompt = ""

        image = image.astype(np.float32) / 127.5 - 1.
        depth = depth.astype(np.float32) / 255.
        depth = depth.reshape(self.image_size, self.image_size, 1)
        
        return dict(
            txt=prompt,
            jpg=image,
            hint=depth
        )
    
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
        # pil_image = center_crop_and_resize(pil_image, (self.image_size, self.image_size), Image.Resampling.BICUBIC)
        image = np.array(pil_image)

        pil_seg_image = Image.open(row.seg_path).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        # pil_seg_image = center_crop_and_resize(pil_seg_image, (self.image_size, self.image_size), Image.Resampling.NEAREST)
        seg_image = self.ade20k_palette[np.array(pil_seg_image)].astype(np.uint8)
        canny = self.canny_detector(seg_image, 100, 200)
        prompt = row.caption
        if random.random() < self.p_drop:
            prompt = ""

        image = image.astype(np.float32) / 127.5 - 1.
        canny = canny.astype(np.float32) / 255.
        canny = canny.reshape(self.image_size, self.image_size, 1)
        
        return dict(
            txt=prompt,
            jpg=image,
            hint=canny
        )

class SemanticSegmentationControl(Dataset):
    
    def __init__(self, metadata_path, image_size, p_drop=0.5):
        super().__init__()
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        self.image_size = image_size
        self.p_drop = p_drop

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        pil_image = Image.open(row.path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        # pil_image = center_crop_and_resize(pil_image, (self.image_size, self.image_size), Image.Resampling.BICUBIC)
        image = np.array(pil_image)

        pil_seg_image = Image.open(row.seg_path).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        # pil_seg_image = center_crop_and_resize(pil_seg_image, (self.image_size, self.image_size), Image.Resampling.NEAREST)
        seg_image = np.array(pil_seg_image)
        
        prompt = row.caption
        if random.random() < self.p_drop:
            prompt = ""

        image = image.astype(np.float32) / 127.5 - 1.
        seg_image = seg_image.reshape(self.image_size, self.image_size, 1).astype(np.float32) / 255.
        
        return dict(
            txt=prompt,
            jpg=image,
            hint=seg_image
        )

class PoseControl(Dataset):
    def __init__(self, root, metadata_path, keypoints_json_path, image_size, p_drop=0.5):
        super().__init__()
        self.root = root
        self.metadata_path = metadata_path
        self.keypoints_json_path = keypoints_json_path
        self.metadata = pd.read_csv(metadata_path)
        with open(self.keypoints_json_path, "r") as f:
            self.keypoints_data = json.load(f)

        image_id_to_objects = defaultdict(list)
        for object_anno in self.keypoints_data["annotations"]:
            object_anno.pop("segmentation", None)
            object_anno.pop("area", None)
            object_anno.pop("iscrowd", None)
            object_anno.pop("id", None)
            image_id = object_anno['image_id']
            image_id_to_objects[image_id].append(object_anno)
        
        self.image_id_to_objects = image_id_to_objects
        
        self.image_size = image_size
        self.p_drop = p_drop

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = int(row["id"])
        pil_image = Image.open(os.path.join(self.root, row.path)).convert("RGB") 
        width, height = pil_image.size
        pil_image = pil_image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        # pil_image = center_crop_and_resize(pil_image, (self.image_size, self.image_size))
        image = np.array(pil_image)

        pil_pose_image = Image.open(os.path.join(self.root, row.pose_path)).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        # pil_pose_image = center_crop_and_resize(pil_pose_image, (self.image_size, self.image_size))
        pose_image = np.array(pil_pose_image)
        
        prompt = row.caption
        if random.random() < self.p_drop:
            prompt = ""

        image = image.astype(np.float32) / 127.5 - 1.
        pose_image = pose_image.astype(np.float32) / 255.

        object_annos = self.image_id_to_objects[image_id]
        gt_detections = []
        for anno_idx, object_anno in enumerate(object_annos):
            keypoints = np.array(object_anno['keypoints']).reshape((17, 3))
            if np.all(keypoints[:, 2] != 2):
                continue
            
            keypoints[:,0] = keypoints[:,0] * 512 / width
            keypoints[:,1] = keypoints[:,1] * 512 / height
            gt_detections.append(object_anno.copy())
            gt_detections[-1]["image_id"] = 1
            gt_detections[-1]["id"] = anno_idx + 1
            gt_detections[-1]["iscrowd"] = 0
            gt_detections[-1]["area"] = object_anno["bbox"][2]*object_anno["bbox"][3]
            gt_detections[-1]["keypoints"] = keypoints.flatten().tolist()

        gt_anno = dict(
            # info=self.keypoints_data["info"],
            images=[{"id": 1}],
            annotations=gt_detections,
            categories=self.keypoints_data["categories"]
        )        
        
        return dict(
            txt=prompt,
            jpg=image,
            hint=pose_image,
            detections=gt_anno
        )