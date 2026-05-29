import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
from PIL import Image, PngImagePlugin
import einops
import cv2
import os
from collections import defaultdict
import random

LARGE_ENOUGH_NUMBER = 1024
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)


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
        image = np.array(pil_image)

        pil_pose_image = Image.open(os.path.join(self.root, row.pose_path)).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        pose = np.array(pil_pose_image)
        image = einops.rearrange(torch.from_numpy(image.astype(np.float32) / 255.), "h w c ->  c h w")
        pose = einops.rearrange(torch.from_numpy(pose.astype(np.float32) / 255.), "h w c ->  c h w")

        sentence = row.caption
        if random.random() < self.p_drop:
            sentence = ""


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


        return {'im': image, 'pose': pose, 'sentence': sentence, 'gt': gt_anno}
