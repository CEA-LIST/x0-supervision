import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import seed_everything
import einops
import numpy as np
import pandas as pd
from PIL import Image
from utils.control_evaluators import PoseEvaluator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from cldm.model import create_model, load_state_dict
import json
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
import os
from  argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--metadata_path", type=str)
parser.add_argument("--keypoints_json", type=str)
parser.add_argument("--image_size", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument('--ddim_steps', type=int, default=50)
parser.add_argument('--ddim_eta', type=float, default=0.)
parser.add_argument("--cfg_scale", type=float, default=9.)
parser.add_argument('--negative_prompts', type=str, default="lowres, cropped, worst quality, low quality, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured")
parser.add_argument("--save_path", type=str)
parser.add_argument("--seed", type=int, default=None)

## Utils

def center_crop_and_resize(image, image_size):
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
    resized_img = cropped_img.resize((image_size[1], image_size[0]), Image.Resampling.LANCZOS)

    return resized_img

class EvalData(Dataset):
    def __init__(self, root, metadata_path, keypoints_json_path, image_size):
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
    
def collate_fn(batch):
    txts = [sample['txt'] for sample in batch]
    keys = set(batch[0].keys()) - {'txt'}
    res = dict(txt=txts)
    for key in keys:
        if isinstance(batch[0][key], np.ndarray):
            res[key] = torch.stack([torch.FloatTensor(sample[key]) for sample in batch])
        else:
            res[key] = [sample[key] for sample in batch]
    return res

##


if __name__ == "__main__":
    args = parser.parse_args()
    if args.seed:
        print(f"Setting seed to {args.seed}")
        seed_everything(args.seed)

    print(f"Creating the model")
    model = create_model(args.config).cpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Loading model weights ...")
    missing, unexpected = model.load_state_dict(load_state_dict(args.ckpt_path, location='cpu'), strict=False)
    print(f"Restored from {args.ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0:
        print(f"Missing Keys:\n {missing}")
    if len(unexpected) > 0:
        print(f"\nUnexpected Keys:\n {unexpected}")

    dataset = EvalData(
        root=args.data_path,
        metadata_path=args.metadata_path,
        keypoints_json_path=args.keypoints_json,
        image_size=args.image_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False
    )

    control_evaluator = PoseEvaluator()
    IS = InceptionScore()
    FID = FrechetInceptionDistance(feature=2048)

    save_folder = Path(args.save_path)
    save_folder.mkdir(exist_ok=True, parents=True)

    metadata = dict(
        config=dict(
            model=model.__class__.__name__,
            data_path=args.data_path,
            ckpt_path=args.ckpt_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cfg_scale=args.cfg_scale,
            ddim_steps=args.ddim_steps,
            negative_prompt=args.negative_prompts
        ),
        samples=[],
        overall_results=dict()
    )

    use_ddim = args.ddim_steps is not None
    results = {k: [] for k in control_evaluator.metrics}
    global_idx = 0
    for batch in tqdm(dataloader):
        # batch_filenames = batch['filename']
        images = (einops.rearrange(batch['jpg'], 'b h w c -> b c h w') * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        FID.update(images, real=True)
        prompts = batch['txt']
        controls = batch['hint'].to(device)
        controls = einops.rearrange(controls, 'b h w c -> b c h w')
        controls = controls.to(memory_format=torch.contiguous_format)
        bs = controls.size(0)

        gt_detections = batch["detections"]

        text_embedding = model.get_learned_conditioning(prompts)
        c = dict(c_concat=[controls], c_crossattn=[text_embedding])
        sampling_kwargs = dict(
            cond=c,
            batch_size=bs,
            ddim=use_ddim,
            ddim_steps=args.ddim_steps,
            eta=args.ddim_eta,
        )

        if args.cfg_scale > 1.0:
            negative_prompt = args.negative_prompts
            uc_cross = model.get_learned_conditioning([negative_prompt]*bs)
            uc = dict(c_concat=[controls], c_crossattn=[uc_cross])
            sampling_kwargs['unconditional_guidance_scale'] = args.cfg_scale
            sampling_kwargs['unconditional_conditioning'] = uc

        with torch.no_grad():
            samples, _ = model.sample_log(**sampling_kwargs)
        
        generated_images = (torch.clamp(model.decode_first_stage(samples).cpu(), -1.0, 1.0) + 1.0) / 2.0
        generated_images = (generated_images * 255).to(torch.uint8)
        FID.update(generated_images, real=False)
        IS.update(generated_images)

        generated_images = einops.rearrange(generated_images, "b c h w -> b h w c").numpy()
        # generated_images = np.ascontiguousarray((torch.clamp(generated_images, -1., 1.)*127.5 + 127.5).permute(0, 2, 3, 1).numpy().astype(np.uint8))
        generated_images = [Image.fromarray(im) for im in generated_images]
        for idx, generated_image in enumerate(generated_images):
            generated_image.save(save_folder / f"{global_idx+idx}.jpg")

        for idx, gt_detection in enumerate(gt_detections):
            with (save_folder / f"{global_idx+idx}.json").open('w') as f:
                json.dump(gt_detection, f, indent=4)

        global_idx += bs

        batch_results = control_evaluator(generated_images, gt_detections)
        for k in batch_results.keys():
            results[k].extend(batch_results[k])

    for k in results.keys():
        metadata['overall_results'][k] = np.nanmean(results[k])

    metadata['overall_results']['FID'] = FID.compute().item()
    IS_mean, IS_std = IS.compute()
    metadata['overall_results']['IS'] = IS_mean.item()
    metadata['overall_results']['IS_std'] = IS_std.item()

    json.dump(metadata, (save_folder / 'metadata.json').open("w"), indent=4)