# Instructions for T2I-Adapter

## Installation
Move to the T2I-Adapter folder:
```sh
cd T2I-Adapter
```
Then run:
```
uv sync --index-strategy unsafe-best-match
```
There is an import error in the version of the `basicsr` package used in this project. Run this command to remove it:
```sh
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' .venv/lib/python3.10/site-packages/basicsr/data/degradations.py
```

## Training
First download the [Stable Diffusion 1.5 checkpoint](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt) and place it in the [checkpoints](./checkpoints) folder.

A specific configuration file is created for each supervision approach. Each supervision has a config file per task. Find them in the [configs/stable-diffusion](./configs/stable-diffusion) folder.


### Depth
```sh
uv run train_depth.py \
    --data_root <path_to_root> \
	--p_text_drop 0.5 \
	--gpus 1 \
	--config configs/stable-diffusion/<config_file> \
	--ckpt checkpoints/v1-5-pruned.ckpt \
	--epochs 10000 \
	--num_workers 8 \
	--bsize 8
```


### Semantic Segmentation
```sh
uv run train_seg.py \
	--gpus 1 \
	--config configs/stable-diffusion/<config_file> \
	--ckpt checkpoints/v1-5-pruned.ckpt \
	--epochs 10000 \
	--num_workers 8 \
	--bsize 8 \
    --seed 1337
```

### Canny Edge
```sh
uv run train_canny.py \
    --config configs/stable-diffusion/<config_file> \
	--ckpt checkpoints/v1-5-pruned.ckpt \
	--train_metadata data/ade20k_train_caption.csv \
	--val_metadata data/ade20k_val_caption.csv \
	--p_drop_text 0.5 \
	--epochs 10000 \
	--gpus 8 \
	--bsize 8 \
	--num_workers 8 \
	--seed 1337
```

### Pose

```sh
uv run train_pose.py \
    --config configs/stable-diffusion/<config_file> \
	--ckpt checkpoints/v1-5-pruned.ckpt \
	--gpus 8 \
	--train_data_root <path_to_parent_directory>/ControlNet/control_datasets/coco_keypoints/train \
	--val_data_root <path_to_parent_directory>/ControlNet/control_datasets/coco_keypoints/val \
	--train_metadata_path <path_to_parent_directory>/ControlNet/control_datasets/coco_keypoints/train/metadata.csv \
	--val_metadata_path <path_to_parent_directory>/control_datasets/coco_keypoints/val/metadata.csv \
	--train_json_path <path_to_coco2017_folder>/annotations/person_keypoints_train2017.json \
	--val_json_path <path_to_coco2017_folder>/annotations/person_keypoints_val2017.json \
	--p_text_drop 0.5 \
	--epochs 10000 \
	--num_workers 8 \
	--bsize 8 \
    --seed 1337
```

## Evaluation
To generate and evaluate the control models use the followings

### Depth
```sh
uv run generate_depth_eval_samples.py \
    --config configs/stable-diffusion/<config_file> \
	--sd_ckpt_path checkpoints/v1-5-pruned.ckpt \
	--ad_ckpt_path experiments/<path_to_checkpoint> \
	--data_path <path_to_depth_data> \
	--image_size 512 \
	--batch_size 1 \
	--num_workers 8 \
	--ddim_steps 50 \
	--cfg_scale 7.5 \
	--save_path <save_path> \
	--seed 123
```

### Semantic Segmentation
```sh
uv run generate_seg_eval_samples.py \
    --config configs/stable-diffusion/<config_file> \
	--sd_ckpt_path checkpoints/v1-5-pruned.ckpt \
	--ad_ckpt_path experiments/<path_to_checkpoint> \
	--metadata_path data/ade20k_val_caption.csv \
	--image_size 512 \
	--batch_size 1 \
	--ddim_steps 50 \
	--cfg_scale 7.5 \
	--save_path <save_path> \
	--seed 123
```

### Canny Edge
```sh
uv run generate_canny_eval_samples.py \
    --config configs/stable-diffusion/<config_file> \
	--sd_ckpt_path checkpoints/v1-5-pruned.ckpt \
	--ad_ckpt_path experiments/<path_to_checkpoint> \
	--metadata_path data/ade20k_val_caption.csv \
	--image_size 512 \
	--batch_size 1 \
	--ddim_steps 50 \
	--cfg_scale 7.5 \
	--save_path <save_path> \
	--seed 123
```

### Pose
```sh
uv run generate_pose_eval_samples.py \
    --config configs/stable-diffusion/<config_file> \
	--sd_ckpt_path checkpoints/v1-5-pruned.ckpt \
	--ad_ckpt_path experiments/<path_to_checkpoint> \
	--data_path <path_to_parent_directory>/ControlNet/control_datasets/coco_keypoints/val \
	--metadata_path <path_to_parent_directory>/ControlNet/control_datasets/coco_keypoints/val/metadata.csv \
	--json_path <path_to_coco2017_folder>/annotations/person_keypoints_val2017.json \
	--image_size 512 \
	--batch_size 1 \
	--num_workers 8 \
	--ddim_steps 50 \
	--cfg_scale 7.5 \
	--save_path <save_path> \
	--seed 123
```
