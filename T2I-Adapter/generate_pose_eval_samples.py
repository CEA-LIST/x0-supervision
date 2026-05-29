import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import seed_everything
import einops
import numpy as np
import pandas as pd
import cv2
import csv
from basicsr.utils import img2tensor
from ldm.util import instantiate_from_config
from ldm.modules.encoders.adapter import Adapter
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image, PngImagePlugin
from utils.pose_evaluator import PoseEvaluator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from omegaconf import OmegaConf
from datasets import load_from_disk
import json
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
import os
from  argparse import ArgumentParser

LARGE_ENOUGH_NUMBER = 1024
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

parser = ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--sd_ckpt_path", type=str)
parser.add_argument("--ad_ckpt_path", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--metadata_path", type=str)
parser.add_argument("--json_path", type=str)
parser.add_argument("--image_size", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument('--ddim_steps', type=int, default=50)
parser.add_argument('--ddim_eta', type=float, default=0.)
parser.add_argument("--cfg_scale", type=float, default=9.)
parser.add_argument('--negative_prompts', type=str, default="lowres, cropped, worst quality, low quality, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured")
parser.add_argument("--save_path", type=str)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--C", type=int, default=4)
parser.add_argument("--f", type=int, default=8)

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
        image = np.array(pil_image)

        pil_pose_image = Image.open(os.path.join(self.root, row.pose_path)).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        pose = np.array(pil_pose_image)
        image = einops.rearrange(torch.from_numpy(image.astype(np.float32) / 255.), "h w c ->  c h w")
        pose = einops.rearrange(torch.from_numpy(pose.astype(np.float32) / 255.), "h w c ->  c h w")

        sentence = row.caption

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
    
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def collate_fn(batch):
    txts = [sample['sentence'] for sample in batch]
    keys = set(batch[0].keys()) - {'sentence'}
    res = dict(sentence=txts)
    for key in keys:
        if isinstance(batch[0][key], np.ndarray):
            res[key] = torch.stack([torch.FloatTensor(sample[key]) for sample in batch])
        elif isinstance(batch[0][key], torch.Tensor):
            res[key] = torch.stack([sample[key] for sample in batch])
        else:
            res[key] = [sample[key] for sample in batch]
    return res

if __name__ == "__main__":
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.seed:
        print(f"Setting seed to {args.seed}")
        seed_everything(args.seed)

    print(f"Creating the model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_config(config, args.sd_ckpt_path).to(device)

    model_ad = Adapter(cin=int(3*64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)
    missing, unexpected = model_ad.load_state_dict(load_state_dict(args.ad_ckpt_path, location='cpu'), strict=False)
    model_ad = model_ad.to(device)

    sampler = DDIMSampler(model)

    dataset = EvalData(
        root=args.data_path,
        metadata_path=args.metadata_path,
        keypoints_json_path=args.json_path,
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
            metadata_path=args.metadata_path,
            json_path=args.json_path,
            sd_ckpt_path=args.sd_ckpt_path,
            ad_ckpt_path=args.ad_ckpt_path,
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

    genImages = []
    gts = []
    id = 0
    results = {k: [] for k in control_evaluator.metrics}
    # count = 0
    for batch in tqdm(dataloader):
        images = (batch['im'] * 255).clamp(0, 255).to(torch.uint8)
        FID.update(images, real=True)
        c = model.get_learned_conditioning(batch['sentence'])
        pose = batch['pose']
        bs = pose.size(0)

        features_adapter = model_ad(pose.to(device))
        shape = [args.C, args.image_size // args.f, args.image_size // args.f]
        with torch.no_grad():
            samples, _ = sampler.sample(S=args.ddim_steps,
                                                conditioning=c,
                                                batch_size=args.batch_size,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=args.cfg_scale,
                                                unconditional_conditioning=model.get_learned_conditioning([args.negative_prompts]*bs),
                                                eta=args.ddim_eta,
                                                x_T=None,
                                                features_adapter=features_adapter)
        generated_images = (torch.clamp(model.decode_first_stage(samples).cpu(), -1.0, 1.0) + 1.0) / 2.0
        generated_images = (generated_images * 255).to(torch.uint8)
        FID.update(generated_images, real=False)
        IS.update(generated_images)

        generated_images = einops.rearrange(generated_images, "b c h w -> b h w c").numpy()
        generated_images = [Image.fromarray(im) for im in generated_images]
        gts = batch['gt']
        # genImages.extend(generated_images)
        batch_results = control_evaluator(generated_images, gts)
        
        for k in batch_results.keys():
            results[k].extend(batch_results[k])

        for idx in range(bs):
            sample_dict = dict()
            generated_image = generated_images[idx]
            image_id = id
            id += 1
            filename = (save_folder / f"generated_{image_id}.jpg").as_posix()
            generated_image.save(filename)
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