import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import einops
from datasets import load_from_disk
import numpy as np
import pandas as pd
import cv2
from annotator.canny import CannyDetector
from PIL import Image, PngImagePlugin
import json
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from typing import Union, Tuple, Optional
import os
import random

LARGE_ENOUGH_NUMBER = 1024
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

    
class DepthControl(Dataset):
    
    def __init__(self, root, split, condition_size, target_size, p_drop=0.5, position_scale=1.0):
        super().__init__()
        self.root = root
        self.split = split
        self.base_dataset = load_from_disk(self.root)[self.split]
        self.condition_size = condition_size
        self.target_size = target_size
        self.p_drop = p_drop
        self.to_tensor = T.ToTensor()
        self.position_scale = position_scale

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        row = self.base_dataset[idx]
        image = row["image"].resize(self.target_size, Image.Resampling.BICUBIC).convert("RGB")
        depth = row["control_depth"].resize(self.condition_size, Image.Resampling.NEAREST).convert("RGB")
        gt = np.array(row["control_depth"].resize(self.condition_size, Image.Resampling.NEAREST).convert("L")).astype(np.float32)
        gt = einops.rearrange(torch.from_numpy(gt), "h w -> 1 h w")
        position_delta = np.array([0, 0])
        position_scale = self.position_scale

        description = row["text"]
        if random.random() < self.p_drop:
            description = ""
        
        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(depth),
            "condition_type_0": "depth",
            "position_delta_0": position_delta,
            "description": description,
            "gt": gt,
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }
    
class CannyControl(Dataset):
    
    def __init__(self, metadata_path, condition_size, target_size, p_drop=0.5, position_scale=1.0):
        super().__init__()
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        self.condition_size = condition_size
        self.target_size = target_size
        self.p_drop = p_drop
        self.to_tensor = T.ToTensor()
        self.position_scale = position_scale
        self.ade20k_palette = np.load("ade20k_palette.npy")
        self.canny_detector = CannyDetector()

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image = Image.open(row.path).convert("RGB").resize(self.target_size, Image.Resampling.BICUBIC)

        pil_seg_image = Image.open(row.seg_path).resize(self.condition_size, Image.Resampling.NEAREST)
        seg_image = self.ade20k_palette[np.array(pil_seg_image)].astype(np.uint8)
        gt = self.canny_detector(seg_image, 100, 200)
        canny = Image.fromarray(gt).convert("RGB")
        gt = einops.rearrange(torch.from_numpy(gt.astype(np.float32)), "h w -> 1 h w")
        position_delta = np.array([0, 0])
        position_scale = self.position_scale
        description = row.caption
        if random.random() < self.p_drop:
            description = ""

        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(canny),
            "condition_type_0": "canny",
            "position_delta_0": position_delta,
            "description": description,
            "gt": gt,
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }
    

class SemanticSegmentationControl(Dataset):
    
    def __init__(self, metadata_path, condition_size, target_size, p_drop=0.5, position_scale=1.0):
        super().__init__()
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        self.condition_size = condition_size
        self.target_size = target_size
        self.p_drop = p_drop
        self.to_tensor = T.ToTensor()
        self.position_scale = position_scale

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image = Image.open(row.path).convert("RGB").resize(self.target_size, Image.Resampling.BICUBIC)
        seg = Image.open(row.seg_path).resize(self.condition_size, Image.Resampling.NEAREST).convert("RGB")
        gt = np.array(Image.open(row.seg_path).resize(self.condition_size, Image.Resampling.NEAREST).convert("L")).astype(np.float32)
        gt = einops.rearrange(torch.from_numpy(gt), "h w -> 1 h w")
        position_delta = np.array([0, 0])
        position_scale = self.position_scale

        description = row.caption
        if random.random() < self.p_drop:
            description = ""
        
        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(seg),
            "condition_type_0": "depth",
            "position_delta_0": position_delta,
            "description": description,
            "gt": gt,
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }

class PoseControl(Dataset):
    def __init__(self, root, metadata_path, keypoints_json_path, condition_size, target_size, p_drop=0.5, position_scale=1.0):
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
        
        self.condition_size = condition_size
        self.target_size = target_size
        self.p_drop = p_drop
        self.to_tensor = T.ToTensor()
        self.position_scale = position_scale

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = int(row["id"])
        image = Image.open(os.path.join(self.root, row.path)).convert("RGB")
        width, height = image.size
        image = image.resize(self.target_size, Image.Resampling.BICUBIC)
        pose= Image.open(os.path.join(self.root, row.pose_path)).convert("RGB").resize(self.condition_size, Image.Resampling.NEAREST)
        position_delta = np.array([0, 0])
        position_scale = self.position_scale
        
        description = row.caption
        if random.random() < self.p_drop:
            description = ""

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

        gt = dict(
            images=[{"id": 1}],
            annotations=gt_detections,
            categories=self.keypoints_data["categories"]
        )        
        
        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(pose),
            "condition_type_0": "depth",
            "position_delta_0": position_delta,
            "description": description,
            "gt": gt,
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }