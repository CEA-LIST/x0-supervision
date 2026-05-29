# Instructions for OminiControl

## Installation
Move to the OminiControl folder:
```sh
cd OminiControl
```
Then run:
```
uv sync --index-strategy unsafe-best-match
```

## Training

Replace `<config_file>` withe the correct config file in [train/config](./train/config/) folder and run:

```sh
export OMINI_CONFIG=./train/config/<config_file>
uv run accelerate launch --main_process_port 41353 -m omini.train_flux.train_spatial_alignment_v2
```

## Evaluation
To generate and evaluate the control models use the followings

### Depth
```sh
uv run python generate_depth_eval_samples.py \
  --ckpt_path runs/<path_to_ckpt_file> \
  --data_path <path_to_depth_data_path> \
  --image_size 512 \
  --save_path <save_path>
```

### Semantic Segmentation
```sh
uv run generate_seg_eval_samples.py \
  --ckpt_path runs/<path_to_ckpt_file> \
  --input_csv <path_to_parent_directory>/ControlNet/ade20k_val_caption.csv \
  --image_size 512 \
  --save_path <save_path> \
  --seed 123
```

### Canny Edge
```sh
uv run generate_canny_eval_samples.py \
  --ckpt_path runs/<path_to_ckpt_file> \
  --input_csv <path_to_parent_directory>/ControlNet/ade20k_val_caption.csv \
  --image_size 512 \
  --save_path <save_path> \
  --seed 123
```

### Pose
```sh
uv run generate_pose_eval_samples.py \
  --ckpt_path runs/<path_to_ckpt_file> \
  --data_path <path_to_parent_directory>/ControlNet/control_datasets/coco_keypoints/val \
  --metadata_path <path_to_parent_directory>/ControlNet/control_datasets/coco_keypoints/val/metadata.csv \
  --keypoints_json <path_to_parent_directory>/ControlNet/person_keypoints_val2017.json \
  --image_size 512 \
  --save_path <save_path> \
  --seed 123
```
