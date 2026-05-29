import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .utils import collate_fn
import einops
import torchvision
from PIL import Image
from pathlib import Path
import json
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from cldm.model import create_control_evaluator, create_eval_dataset
from tqdm.auto import tqdm

NEGATIVE_PROMPT = "lowres, cropped, worst quality, low quality, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"

class ControlFidelityEvalCallback(Callback):
    def __init__(self,
                 config_path,
                 batch_frequency=2000,
                 num_eval_images=1024,
                 eval_batch_size=8,
                 eval_num_workers=8,
                 cfg_scale=8.,
                 negative_prompt=NEGATIVE_PROMPT,
                 ddim_steps=50,
                 ddim_eta=0.):
        super().__init__()
        self.batch_freq = batch_frequency
        self.num_eval_images = num_eval_images
        self.eval_batch_size = eval_batch_size
        eval_dataset = create_eval_dataset(config_path)
        step = max(len(eval_dataset) // num_eval_images, 1)
        eval_dataset = Subset(eval_dataset, list(range(0, len(eval_dataset), step)))
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=eval_num_workers
        )
        self.evaluator = None
        self.cfg_scale = cfg_scale
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.negative_prompt = negative_prompt
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.instantiate_control_evaluator(config_path)

    @rank_zero_only
    def instantiate_control_evaluator(self, config_path):
        self.evaluator = create_control_evaluator(config_path)

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0
    
    @torch.no_grad()
    def eval(self, pl_module):
        assert self.evaluator is not None
        print("Starting model evaluation ...")
        save_dir = Path(f"{pl_module.logger.experiment.get_logdir()}/iter_{pl_module.global_step}_eval_samples")
        save_dir.mkdir(exist_ok=True, parents=True)
        (save_dir / "samples").mkdir(exist_ok=True)
        (save_dir / "controls").mkdir(exist_ok=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            results = {k: [] for k in self.evaluator.metrics}
            global_idx = 0
            for batch in tqdm(self.eval_dataloader):
                prompts = batch['txt']
                controls = batch['hint'].to(self.device_type, torch.float16)
                controls = einops.rearrange(controls, 'b h w c -> b c h w')
                controls = controls.to(memory_format=torch.contiguous_format)
                bs = controls.size(0)

                text_embedding = pl_module.get_learned_conditioning(prompts)
                c = dict(c_concat=[controls], c_crossattn=[text_embedding])
                sampling_kwargs = dict(
                    cond=c,
                    batch_size=bs,
                    ddim=self.ddim_steps is not None,
                    ddim_steps=self.ddim_steps,
                    eta=self.ddim_eta,
                )

                if self.cfg_scale > 1.0:
                    negative_prompt = self.negative_prompt
                    uc_cross = pl_module.get_learned_conditioning([negative_prompt]*bs)
                    uc = dict(c_concat=[controls], c_crossattn=[uc_cross])
                    sampling_kwargs['unconditional_guidance_scale'] = self.cfg_scale
                    sampling_kwargs['unconditional_conditioning'] = uc

                samples, _ = pl_module.sample_log(**sampling_kwargs)
                generated_images = pl_module.decode_first_stage(samples).cpu()
                generated_images = np.ascontiguousarray((einops.rearrange(torch.clamp(generated_images*127.5 + 127.5, 0., 255.), 'b c h w -> b h w c')).numpy().astype(np.uint8))
                generated_images = [Image.fromarray(im) for im in generated_images]

                for idx, generated_image in enumerate(generated_images):
                    generated_image.save(save_dir / f"samples/{global_idx+idx}.jpg")

                controls = torch.clamp(einops.rearrange(controls, 'b c h w -> b h w c') * 255., 0., 255.).cpu().squeeze(-1).numpy().astype(np.uint8)

                for idx, control in enumerate(controls):
                    Image.fromarray(control).save(save_dir / f"controls/{global_idx+idx}.png")

                global_idx += bs

                batch_results = self.evaluator(generated_images, controls)
                for k in batch_results.keys():
                    results[k].extend(batch_results[k])
            
            for k in results.keys():
                pl_module.logger.experiment.add_scalar(k, np.nanmean(results[k]), pl_module.global_step)

        print("Evaluation completed. Moving on to training ...")

    @rank_zero_only
    def launch_eval(self, model):
        model.eval()
        self.eval(model)
        model.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        check_idx = pl_module.global_step
        if self.check_frequency(check_idx):
            self.launch_eval(pl_module)


class ControlFidelityEvalCallbackForDetection(Callback):
    def __init__(self,
                 config_path,
                 batch_frequency=2000,
                 num_eval_images=1024,
                 eval_batch_size=8,
                 eval_num_workers=8,
                 cfg_scale=8.,
                 negative_prompt=NEGATIVE_PROMPT,
                 ddim_steps=50,
                 ddim_eta=0.):
        super().__init__()
        self.batch_freq = batch_frequency
        self.num_eval_images = num_eval_images
        self.eval_batch_size = eval_batch_size
        eval_dataset = create_eval_dataset(config_path)
        step = max(len(eval_dataset) // num_eval_images, 1)
        eval_dataset = Subset(eval_dataset, list(range(0, len(eval_dataset), step)))
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=eval_num_workers
        )
        self.evaluator = None
        self.cfg_scale = cfg_scale
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.negative_prompt = negative_prompt
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.instantiate_control_evaluator(config_path)

    @rank_zero_only
    def instantiate_control_evaluator(self, config_path):
        self.evaluator = create_control_evaluator(config_path)

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0
    
    @torch.no_grad()
    def eval(self, pl_module):
        assert self.evaluator is not None
        print("Starting model evaluation ...")
        save_dir = Path(f"{pl_module.logger.experiment.get_logdir()}/iter_{pl_module.global_step}_eval_samples")
        save_dir.mkdir(exist_ok=True, parents=True)
        (save_dir / "samples").mkdir(exist_ok=True)
        (save_dir / "controls").mkdir(exist_ok=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            results = {k: [] for k in self.evaluator.metrics}
            global_idx = 0
            for batch in tqdm(self.eval_dataloader):
                prompts = batch['txt']
                controls = batch['hint'].to(self.device_type, torch.float16)
                controls = einops.rearrange(controls, 'b h w c -> b c h w')
                controls = controls.to(memory_format=torch.contiguous_format)
                bs = controls.size(0)

                gt_detections = batch["detections"]

                text_embedding = pl_module.get_learned_conditioning(prompts)
                c = dict(c_concat=[controls], c_crossattn=[text_embedding])
                sampling_kwargs = dict(
                    cond=c,
                    batch_size=bs,
                    ddim=self.ddim_steps is not None,
                    ddim_steps=self.ddim_steps,
                    eta=self.ddim_eta,
                )

                if self.cfg_scale > 1.0:
                    negative_prompt = self.negative_prompt
                    uc_cross = pl_module.get_learned_conditioning([negative_prompt]*bs)
                    uc = dict(c_concat=[controls], c_crossattn=[uc_cross])
                    sampling_kwargs['unconditional_guidance_scale'] = self.cfg_scale
                    sampling_kwargs['unconditional_conditioning'] = uc

                samples, _ = pl_module.sample_log(**sampling_kwargs)
                generated_images = pl_module.decode_first_stage(samples).cpu()
                generated_images = np.ascontiguousarray((einops.rearrange(torch.clamp(generated_images*127.5 + 127.5, 0., 255.), 'b c h w -> b h w c')).numpy().astype(np.uint8))
                generated_images = [Image.fromarray(im) for im in generated_images]

                for idx, generated_image in enumerate(generated_images):
                    generated_image.save(save_dir / f"samples/{global_idx+idx}.jpg")

                controls = torch.clamp(einops.rearrange(controls, 'b c h w -> b h w c') * 255., 0., 255.).cpu().squeeze(-1).numpy().astype(np.uint8)

                for idx, control in enumerate(controls):
                    Image.fromarray(control).save(save_dir / f"controls/{global_idx+idx}.png")
                    with open(save_dir / f"controls/{global_idx+idx}.json", "w") as f:
                        json.dump(gt_detections[idx], f, indent=4)
                    
                global_idx += bs

                batch_results = self.evaluator(generated_images, gt_detections)
                for k in batch_results.keys():
                    results[k].extend(batch_results[k])
            
            for k in results.keys():
                pl_module.logger.experiment.add_scalar(k, np.nanmean(results[k]), pl_module.global_step)

        print("Evaluation completed. Moving on to training ...")

    @rank_zero_only
    def launch_eval(self, model):
        model.eval()
        self.eval(model)
        model.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        check_idx = pl_module.global_step
        if self.check_frequency(check_idx):
            self.launch_eval(pl_module)