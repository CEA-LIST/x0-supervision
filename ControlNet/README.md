# Instructions for ControlNet

## Installation
Move to the ControlNet folder:
```sh
cd ControlNet
```
Then run:
```
uv sync --index-strategy unsafe-best-match
```

## Setting up ControlNet for training

To train ControlNet properly, we need to prepare the initial checkpoint in which the weights of the ControlNet blocks are initialized with the those of the original UNet blocks they have been copied from. First download the [Stable Diffusion 1.5 checkpoint](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt) and place it in the [models directory](./models). Then run:

### For depth, canny and semgentation, use this initial checkpoint
```sh
uv run tool_add_control.py configs/controlnet/cldm_v15_depth.yaml models/v1-5-pruned.ckpt models/sd15_control_init1.ckpt
```
### For pose, use this initial checkpoint
```sh
uv run tool_add_control.py configs/controlnet/cldm_v15_pose.yaml models/v1-5-pruned.ckpt models/sd15_control_init2.ckpt
```

## Training
This is for training with a periodic evaluation of the control fidelity on a fixed validation set. The retrieved convergence curve of scores is used to compute the convergence speed. A specific configuration file is created for each supervision approach. The base is $\epsilon$-supervision and those of $v$ and $x_0$ are respectively postfixed by $v$ and $x_0$. Find them in the [configs/controlnet](./configs/controlnet) folder.


### Depth
```sh
uv run train_controlnet_with_periodic_eval.py \
    --config=configs/controlnet/<config_file> \
    --experiment_label=<exp_label> \
    --experiment_version="v0" \
    --pretrained_init_path="models/sd15_control_init1.ckpt" \
    --ckpt_dir=checkpoints \
    --max_steps=200000 \
    --lr=1e-5 \
    --num_eval_images=64 \
    --eval_batch_size=8 \
    --eval_num_workers=8 \
    --eval_freq=1000 \
    --log_freq=1000 \
    --seed=1337
```

### Semantic Segmentation
```sh
uv run train_controlnet_with_periodic_eval.py \
    --config=configs/controlnet/<config_file> \
    --experiment_label=<exp_label> \
    --experiment_version="v0" \
    --pretrained_init_path="models/sd15_control_init1.ckpt" \
    --ckpt_dir=checkpoints \
    --max_steps=200000 \
    --lr=1e-5 \
    --num_eval_images=64 \
    --eval_batch_size=8 \
    --eval_num_workers=8 \
    --eval_freq=1000 \
    --log_freq=1000 \
    --seed=1337
```

### Canny Edge
```sh
uv run train_controlnet_with_periodic_eval.py \
    --config=configs/controlnet/<config_file> \
    --experiment_label=<exp_label> \
    --experiment_version="v0" \
    --pretrained_init_path="models/sd15_control_init1.ckpt" \
    --ckpt_dir=checkpoints \
    --max_steps=200000 \
    --lr=1e-5 \
    --num_eval_images=64 \
    --eval_batch_size=8 \
    --eval_num_workers=8 \
    --eval_freq=1000 \
    --log_freq=1000 \
    --seed=1337
```

### Pose
We need to first prepare the COCO keypoints dataset into the expected format. First run

### Preparing train set
```sh
uv run prepare_coco_keypoints \
    --data_path <path_to_local_coco2017_images_folder> \
    --keypoints_json <path_to_coco2017_folder>/annotations/person_keypoints_train2017.json \
    --captions_json <path_to_coco2017_folder>/annotations/captions_train2017.json \
    --save_folder control_datasets/coco_keypoints/train
```
### Preparing val set
```sh
uv run prepare_coco_keypoints \
    --data_path <path_to_local_coco2017_images_folder> \
    --keypoints_json <path_to_coco2017_folder>/annotations/person_keypoints_val2017.json \
    --captions_json <path_to_coco2017_folder>/annotations/captions_val2017.json \
    --save_folder control_datasets/coco_keypoints/val
```

Note that the same dataset is used for the other methods. So just re-use the same dataset for training the other methods

### Training

```sh
uv run train_controlnet_with_periodic_eval.py \
    --config=configs/controlnet/<config_file> \
    --experiment_label=<exp_label> \
    --experiment_version="v0" \
    --pretrained_init_path="models/sd15_control_init2.ckpt" \
    --ckpt_dir=checkpoints \
    --max_steps=200000 \
    --lr=1e-5 \
    --eval_callback_type="detection" \
    --num_eval_images=64 \
    --eval_batch_size=8 \
    --eval_num_workers=8 \
    --eval_freq=1000 \
    --log_freq=1000 \
    --seed=1337
```

## Evaluation
To generate and evaluate the control models use the followings

### Depth
```sh
uv run generate_depth_eval_samples.py \
    --config configs/controlnet/<config_file> \
    --ckpt_path checkpoints/<ckpt_file> \
    --input_csv imagenet_val_caption.csv \
    --image_size 512 \
    --batch_size 8 \
    --num_workers 8 \
    --ddim_steps 50 \
    --cfg_scale 7.5 \
    --save_path <save_path> \
    --seed 1234
```

### Semantic Segmentation
```sh
uv run generate_seg_eval_samples.py \
    --config configs/controlnet/<config_file> \
    --ckpt_path checkpoints/<ckpt_file> \
    --input_csv ade20k_val_caption.csv \
    --image_size 512 \
    --batch_size 8 \
    --num_workers 8 \
    --ddim_steps 50 \
    --cfg_scale 7.5 \
    --save_path <save_path> \
    --seed 1234
```

### Canny Edge
```sh
uv run generate_canny_eval_samples.py \
    --config configs/controlnet/<config_file> \
    --ckpt_path checkpoints/<ckpt_file> \
    --input_csv ade20k_val_caption.csv \
    --image_size 512 \
    --batch_size 8 \
    --num_workers 8 \
    --ddim_steps 50 \
    --cfg_scale 7.5 \
    --save_path <save_path> \
    --seed 1234
```

### Pose
```sh
uv run generate_pose_eval_samples.py \
    --config configs/controlnet/<config_file> \
    --ckpt_path checkpoints/<ckpt_file> \
    --data_path control_datasets/coco_keypoints/val \
    --metadata_path control_datasets/coco_keypoints/val/metadata.csv \
    --keypoints_json <path_to_coco2017_folder>/annotations/person_keypoints_val2017.json \
    --image_size 512 \
    --batch_size 8 \
    --num_workers 8 \
    --ddim_steps 50 \
    --cfg_scale 7.5 \
    --save_path <save_path> \
    --seed 1234
```
