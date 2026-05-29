
import csv

import numpy as np
from PIL import Image
from basicsr.utils import img2tensor
import cv2


class dataset_ade20k():
    
    def __init__(self, metadata_path):
        super().__init__()
        self.metadata_path = metadata_path
        self.image_size = (512, 512)

        # Read the CSV file using the standard csv API
        self.metadata = []
        with open(self.metadata_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.metadata.append(row)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Access the row as a dictionary from the list
        row = self.metadata[idx]
        
        # Access columns by key instead of attribute
        image_path = row['path']
        seg_path = row['seg_path']
        prompt = row['caption']

        image = cv2.imread(image_path)
        image = cv2.resize(image, self.image_size)
        image = img2tensor(image, bgr2rgb=True, float32=True) / 255.

        seg_image = cv2.imread(seg_path)
        seg_image_int = cv2.resize(seg_image, self.image_size, interpolation=cv2.INTER_NEAREST)
        seg_image = img2tensor(seg_image_int, bgr2rgb=True, float32=True) / 255.
        
        return {'im': image, 'mask': seg_image, 'mask_int': seg_image_int, 'sentence': prompt}
