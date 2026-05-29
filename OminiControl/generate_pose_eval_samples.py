import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import einops
import numpy as np
import pandas as pd
from PIL import Image
from diffusers.pipelines import FluxPipeline
from omini.pipeline.flux_omini import Condition, generate, seed_everything
from omini.pipeline.control_evaluators import PoseEvaluator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import json
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
import os
from  argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--metadata_path", type=str)
parser.add_argument("--keypoints_json", type=str)
parser.add_argument("--image_size", type=int, default=512)
parser.add_argument('--ddim_steps', type=int, default=50)
parser.add_argument('--ddim_eta', type=float, default=0.)
parser.add_argument("--cfg_scale", type=float, default=9.)
parser.add_argument('--negative_prompts', type=str, default="lowres, cropped, worst quality, low quality, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured")
parser.add_argument("--save_path", type=str)
parser.add_argument("--seed", type=int, default=None)

class EvalData(Dataset):
    def __init__(self, root, metadata_path, keypoints_json_path, condition_size, target_size, position_scale=1.0):
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

if __name__ == "__main__":
    args = parser.parse_args()
    if args.seed:
        print(f"Setting seed to {args.seed}")
        seed_everything(args.seed)

    print(f"Creating the model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe = pipe.to(device)
    adapter_name = "default"
    pipe.unload_lora_weights()
    pipe.load_lora_weights(args.ckpt_path, adapter_name=adapter_name)
    pipe.set_adapters([adapter_name], adapter_weights=[1.0])

    dataset = EvalData(
        root=args.data_path,
        metadata_path=args.metadata_path,
        keypoints_json_path=args.keypoints_json,
        condition_size=[args.image_size]*2,
        target_size=[args.image_size]*2
    )

    control_evaluator = PoseEvaluator()
    IS = InceptionScore()
    FID = FrechetInceptionDistance(feature=2048)

    save_folder = Path(args.save_path)
    save_folder.mkdir(exist_ok=True, parents=True)

    metadata = dict(
        config=dict(
            model=pipe.__class__.__name__,
            data_path=args.data_path,
            ckpt_path=args.ckpt_path,
            image_size=args.image_size,
            cfg_scale=args.cfg_scale,
            ddim_steps=args.ddim_steps,
            negative_prompt=args.negative_prompts
        ),
        samples=[],
        overall_results=dict()
    )

    position_delta = [0, 0]
    position_scale = 1.0
    target_size = [512, 512]

    results = {k: [] for k in control_evaluator.metrics}
    id = 0
    # count = 0
    for sample in tqdm(dataset):
        image = sample["image"]
        image = torch.clamp(image * 255, 0, 255).to(torch.uint8)
        FID.update(image.unsqueeze(0), real=True)
        control = sample["condition_0"]
        seg = einops.rearrange(torch.clamp(control * 255, 0, 255), "c h w -> h w c").numpy().astype(np.uint8)[..., 0]
        prompt = sample["description"]
        condition = Condition(control, adapter_name, position_delta, position_scale)
        pil_generated_image = generate(
            pipe,
            prompt=[prompt],
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
        ).images[0]
        
        generated_image = einops.rearrange(torch.from_numpy(np.array(pil_generated_image)).to(torch.uint8), "h w c -> c h w")
        FID.update(generated_image.unsqueeze(0), real=False)
        IS.update(generated_image.unsqueeze(0))

        batch_results = control_evaluator([pil_generated_image], [sample["gt"]])
        for k in batch_results.keys():
            results[k].extend(batch_results[k])

        sample_dict = dict()
        image_id = id
        id += 1
        filename = (save_folder / f"generated_{image_id}.jpg").as_posix()
        pil_generated_image.save(filename)
        sample_dict['id'] = image_id
        sample_dict['filename'] = filename
        metadata['samples'].append(sample_dict)
        
        # count += 1
        # if count == 2: break


    for k in results.keys():
        metadata['overall_results'][k] = np.nanmean(results[k])

    metadata['overall_results']['FID'] = FID.compute().item()
    IS_mean, IS_std = IS.compute()
    metadata['overall_results']['IS'] = IS_mean.item()
    metadata['overall_results']['IS_std'] = IS_std.item()

    json.dump(metadata, (save_folder / 'metadata.json').open("w"), indent=4)