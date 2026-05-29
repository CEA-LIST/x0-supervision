import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import seed_everything
import einops
import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin
from utils.control_evaluators import CannyControlV2Evaluator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from datasets import load_from_disk
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
import json
from pathlib import Path
from tqdm.auto import tqdm
import os
from  argparse import ArgumentParser

LARGE_ENOUGH_NUMBER = 1024
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

parser = ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--input_csv", type=str)
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
    def __init__(self, metadata_path, image_size):
        super().__init__()
        self.metadata_path = metadata_path
        name, ext = os.path.splitext(metadata_path)
        assert ext == ".csv", f"Error: expecting the metadata file to be a .csv file, got '{ext}' instead."
        self.metadata = pd.read_csv(metadata_path)
        self.ade20k_palette = np.load("ade20k_palette.npy")
        self.canny_detector = CannyDetector()
        self.image_size = image_size

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        pil_image = Image.open(row.path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        # pil_image = center_crop_and_resize(pil_image, (self.image_size, self.image_size), Image.Resampling.BICUBIC)
        image = np.array(pil_image)

        pil_seg_image = Image.open(row.seg_path).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        # pil_seg_image = center_crop_and_resize(pil_seg_image, (self.image_size, self.image_size), Image.Resampling.NEAREST)
        seg_image = self.ade20k_palette[np.array(pil_seg_image)].astype(np.uint8)
        canny = self.canny_detector(seg_image, 100, 200)
        prompt = row.caption

        image = image.astype(np.float32) / 127.5 - 1.
        canny = canny.astype(np.float32) / 255.
        canny = canny.reshape(self.image_size, self.image_size, 1)
        
        return dict(
            txt=prompt,
            jpg=image,
            hint=canny
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

    dataset = EvalData(args.input_csv, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False
    )

    control_evaluator = CannyControlV2Evaluator()
    IS = InceptionScore()
    FID = FrechetInceptionDistance(feature=2048)

    save_folder = Path(args.save_path)
    save_folder.mkdir(exist_ok=True, parents=True)
    (save_folder / "samples").mkdir(exist_ok=True)
    (save_folder / "controls").mkdir(exist_ok=True)

    metadata = dict(
        config=dict(
            model=model.__class__.__name__,
            input_csv=args.input_csv,
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
    id = 0
    results = {k: [] for k in control_evaluator.metrics}
    # count = 0
    for batch in tqdm(dataloader):
        # batch_filenames = batch['filename']
        images = (einops.rearrange(batch['jpg'], 'b h w c -> b c h w') * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        FID.update(images, real=True)
        prompts = batch['txt']
        controls = batch['hint'].to(device)
        controls = einops.rearrange(controls, 'b h w c -> b c h w')
        controls = controls.to(memory_format=torch.contiguous_format)
        bs = controls.size(0)

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
        generated_images = model.decode_first_stage(samples).cpu()
        FID.update((generated_images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8), real=False)
        IS.update((generated_images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8))
        generated_images = np.ascontiguousarray((torch.clamp(generated_images, -1., 1.)*127.5 + 127.5).permute(0, 2, 3, 1).numpy().astype(np.uint8))
        generated_images = [Image.fromarray(im) for im in generated_images]
        controls = torch.clamp(einops.rearrange(controls, 'b c h w -> b h w c') * 255., 0., 255.).cpu().squeeze(-1).numpy().astype(np.uint8)
        batch_results = control_evaluator(generated_images, controls)
        for k in batch_results.keys():
            results[k].extend(batch_results[k])

        for idx in range(bs):
            sample_dict = dict()
            generated_image = generated_images[idx]
            control_image = controls[idx]
            # source_filename = batch_filenames[idx]
            image_id = id
            id += 1
            filename = (save_folder / f"samples/{image_id}.jpg").as_posix()
            control_filename = (save_folder / f"controls/{image_id}.png").as_posix()
            generated_image.save(filename)
            Image.fromarray(control_image).save(control_filename)
            sample_dict['id'] = image_id
            sample_dict['filename'] = filename
            sample_dict['control_filename'] = control_filename
            # sample_dict['source_filename'] = source_filename
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