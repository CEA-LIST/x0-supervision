import numpy as np
import pandas as pd
import cv2
import skimage
import json
from tqdm.auto import tqdm
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser

### Utils ###

# This script visualizes COCO-style human pose keypoints.
# The COCO format includes 17 keypoints for a single person.

# Define the 17 COCO keypoints and their standard order.
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Define the connections between keypoints to form the skeleton (limbs).
# Each tuple represents a connection between two keypoint indices from the list above.
SKELETON_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left Arm
    (5, 7), (7, 9),
    # Right Arm
    (6, 8), (8, 10),
    # Left Leg
    (11, 13), (13, 15),
    # Right Leg
    (12, 14), (14, 16)
]

# Define colors for the limbs to replicate the example image.
# The colors are in RGB format, normalized to be between 0 and 1.
LIMB_COLORS = [
    # Head (magenta/purple)
    [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
    # Torso (blue/red)
    [0, 0, 1], [0, 1, 0.5], [1, 0, 0], [0, 1, 0],
    # Left Arm (red/orange)
    [1, 0.5, 0], [1, 0.75, 0],
    # Right Arm (yellow/orange)
    [1, 1, 0], [0.75, 1, 0],
    # Left Leg (blue/cyan)
    [0, 0.5, 1], [0, 0.75, 1],
    # Right Leg (green/cyan)
    [0, 1, 1], [0, 1, 0.75]
]

KEYPOINT_COLORS = [
    # Head
    [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
    # Shoulders
    [1, 0.5, 0], [1, 1, 0],
    # Elbows/Wrists
    [1, 0.75, 0], [0.75, 1, 0], [1, 0.75, 0], [0.75, 1, 0],
    # Hips
    [0, 1, 0.5], [1, 0, 0],
    # Knees/Ankles
    [0, 0.5, 1], [0, 1, 1], [0, 0.75, 1], [0, 1, 0.75]
]


#############

parser = ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--keypoints_json", type=str)
parser.add_argument("--captions_json", type=str)
parser.add_argument("--save_folder", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    save_folder = Path(args.save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)
    (save_folder / "images").mkdir(exist_ok=True)
    (save_folder / "poses").mkdir(exist_ok=True)

    with open(args.keypoints_json, "r") as f:
        keypoints_data = json.load(f)

    with open(args.captions_json, "r") as f:
        captions_data = json.load(f)

    data_folder = Path(args.data_path)
    assert data_folder.exists()

    image_id_to_objects = defaultdict(list)
    for object_anno in keypoints_data["annotations"]:
        object_anno.pop("segmentation", None)
        object_anno.pop("area", None)
        object_anno.pop("iscrowd", None)
        object_anno.pop("id", None)
        image_id = object_anno['image_id']
        image_id_to_objects[image_id].append(object_anno)

    image_id_to_image_info = dict()
    for image_info in keypoints_data["images"]:
        image_id_to_image_info[image_info["id"]] = image_info

    image_id_to_caption = dict()
    for caption_data in captions_data["annotations"]:
        image_id_to_caption[caption_data["image_id"]] = caption_data["caption"]

    image_ids = list(image_id_to_objects.keys())

    samples = dict(
        id=[],
        path=[],
        pose_path=[],
        caption=[]
    )

    for idx in tqdm(range(len(image_ids))):
        image_id = image_ids[idx]
        object_annos = image_id_to_objects[image_id]
        image_info = image_id_to_image_info[image_id]
        caption = image_id_to_caption[image_id]
        height = image_info["height"]
        width = image_info["width"]
        image = skimage.io.imread(data_folder / image_info["file_name"])
        image_size = (512, 512)

        canva = np.zeros((height, width, 3), dtype=np.uint8)
        min_box = np.array([0, 0, width, height])
        visible_persons = 0
        for object_anno in object_annos:
            keypoints = np.array(object_anno['keypoints']).reshape((17, 3))
            if np.all(keypoints[:, 2] != 2):
                continue
            visible_persons += 1
            x, y, w, h = np.array(list(map(lambda x: int(x), object_anno['bbox'])))
            box = np.array([x, y, x+w, y+h])
            if (box[2]-box[0])*(box[3]-box[1]) < (min_box[2]-min_box[0])*(min_box[3]-min_box[1]):
                min_box = box

            # --- Draw the limbs ---
            for i, (p1_idx, p2_idx) in enumerate(SKELETON_CONNECTIONS):
                # Get the start and end points of the limb
                start_point = keypoints[p1_idx]
                end_point = keypoints[p2_idx]

                # Check if both keypoints are visible
                if start_point[2] > 0 and end_point[2] > 0:
                    # Get coordinates
                    x = [start_point[0], end_point[0]]
                    y = [start_point[1], end_point[1]]
                    # Draw the line for the limb
                    color = tuple(map(lambda x: int(x*255), LIMB_COLORS[i]))
                    cv2.line(canva, (start_point[0], start_point[1]), (end_point[0], end_point[1]), color=color, thickness=2)

            # --- Draw the keypoints (joints) ---
            for i, point in enumerate(keypoints):
                # Check if the keypoint is visible
                if point[2] > 0:
                    color = tuple(map(lambda x: int(x*255), KEYPOINT_COLORS[i]))
                    cv2.circle(canva, (point[0], point[1]), radius=5, color=color, thickness=-1)

        min_box[::2] = min_box[::2] * image_size[0] / width
        min_box[1::2] = min_box[1::2] * image_size[1] / height
        min_box_to_image_ratio = ((min_box[2]-min_box[0])*(min_box[3]-min_box[1])) / (image_size[0]*image_size[1])
        if min_box_to_image_ratio < 0.002 or visible_persons > 6 or visible_persons == 0:
            print("No or too small or too much persons, not taking this image")
        else:
            samples["id"].append(image_id)
            image_path = f"images/{image_id}.jpg"
            samples["path"].append(image_path)
            pose_path = f"poses/{image_id}.jpg"
            samples["pose_path"].append(pose_path)
            samples["caption"].append(caption)

            skimage.io.imsave(save_folder / image_path, image)
            skimage.io.imsave(save_folder / pose_path, canva)
    

    df = pd.DataFrame(samples)
    df.to_csv(save_folder / "metadata.csv", index=False)
