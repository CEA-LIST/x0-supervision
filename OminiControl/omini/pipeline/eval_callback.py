import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .utils import collate_fn, instantiate_from_config
import einops
from PIL import Image
from pathlib import Path
import json
from lightning import Callback
from lightning.pytorch.utilities import rank_zero
from tqdm.auto import tqdm
from .flux_omini import Condition, generate

NEGATIVE_PROMPT = "lowres, cropped, worst quality, low quality, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"

class ControlFidelityEvalCallback(Callback):
    def __init__(self,
                 config,
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
        eval_dataset = instantiate_from_config(config["dataset"])
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
        self.instantiate_control_evaluator(config)

    # @rank_zero
    def instantiate_control_evaluator(self, config):
        self.evaluator = instantiate_from_config(config["evaluator"])

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0
    
    @torch.no_grad()
    def eval(self, pl_module, global_step):
        # assert self.evaluator is not None
        print("Starting model evaluation ...")
        save_dir = Path(f"{pl_module.logger.experiment.get_logdir()}/iter_{global_step}_eval_samples")
        save_dir.mkdir(exist_ok=True, parents=True)
        (save_dir / "samples").mkdir(exist_ok=True)
        (save_dir / "controls").mkdir(exist_ok=True)

        # condition_size = pl_module.training_config["dataset"]["params"]["condition_size"]
        target_size = pl_module.training_config["dataset"]["params"]["target_size"]

        position_delta = pl_module.training_config["dataset"]["params"].get("position_delta", [0, 0])
        position_scale = pl_module.training_config["dataset"]["params"].get("position_scale", 1.0)

        adapter = pl_module.adapter_names[2]

        results = {k: [] for k in self.evaluator.metrics}
        global_idx = 0
        for batch in tqdm(self.eval_dataloader):
            prompts = batch['description']
            conditions = [Condition(control, adapter, position_delta, position_scale) for control in batch['condition_0'].to(self.device_type)]
            bs = len(conditions)

            generator = torch.Generator(device=pl_module.device)
            generator.manual_seed(42)
            
            generated_images = []

            for prompt, condition in zip(prompts, conditions):

                res = generate(
                    pl_module.flux_pipe,
                    prompt=prompt,
                    conditions=[condition],
                    height=target_size[1],
                    width=target_size[0],
                    generator=generator,
                    model_config=pl_module.model_config,
                    kv_cache=pl_module.model_config.get("independent_condition", False)
                )

                generated_image = res.images[0]
                generated_images.append(generated_image)

            for idx, generated_image in enumerate(generated_images):
                generated_image.save(save_dir / f"samples/{global_idx+idx}.jpg")

            controls = torch.clamp(einops.rearrange(batch['gt'], 'b c h w -> b h w c'), 0., 255.).cpu().squeeze(-1).numpy().astype(np.uint8)

            for idx, control in enumerate(controls):
                Image.fromarray(control).save(save_dir / f"controls/{global_idx+idx}.png")

            global_idx += bs

            batch_results = self.evaluator(generated_images, controls)
            for k in batch_results.keys():
                results[k].extend(batch_results[k])
        
        for k in results.keys():
            pl_module.logger.experiment.add_scalar(k, np.nanmean(results[k]), global_step)

        print("Evaluation completed. Moving on to training ...")

    # @rank_zero
    def launch_eval(self, model, global_step):
        self.eval(model, global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        is_last_batch_in_accumulation = (batch_idx + 1) % trainer.accumulate_grad_batches == 0
        if is_last_batch_in_accumulation and trainer.global_step % self.batch_freq == 0:
            self.launch_eval(pl_module, trainer.global_step)



class ControlFidelityEvalCallbackForDetection(Callback):
    def __init__(self,
                 config,
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
        eval_dataset = instantiate_from_config(config["dataset"])
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
        self.instantiate_control_evaluator(config)

    # @rank_zero
    def instantiate_control_evaluator(self, config):
        self.evaluator = instantiate_from_config(config["evaluator"])

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0
    
    @torch.no_grad()
    def eval(self, pl_module, global_step):
        assert self.evaluator is not None
        print("Starting model evaluation ...")
        save_dir = Path(f"{pl_module.logger.experiment.get_logdir()}/iter_{global_step}_eval_samples")
        save_dir.mkdir(exist_ok=True, parents=True)
        (save_dir / "samples").mkdir(exist_ok=True)
        (save_dir / "controls").mkdir(exist_ok=True)

        # condition_size = pl_module.training_config["dataset"]["params"]["condition_size"]
        target_size = pl_module.training_config["dataset"]["params"]["target_size"]

        position_delta = pl_module.training_config["dataset"]["params"].get("position_delta", [0, 0])
        position_scale = pl_module.training_config["dataset"]["params"].get("position_scale", 1.0)

        adapter = pl_module.adapter_names[2]

        with torch.cuda.amp.autocast(dtype=torch.float16):
            results = {k: [] for k in self.evaluator.metrics}
            global_idx = 0
            for batch in tqdm(self.eval_dataloader):
                prompts = batch['description']
                conditions = [Condition(control, adapter, position_delta, position_scale) for control in batch['condition_0'].to(self.device_type)]
                bs = len(conditions)

                generator = torch.Generator(device=pl_module.device)
                generator.manual_seed(42)
                
                generated_images = []
                for prompt, condition in zip(prompts, conditions):

                    res = generate(
                        pl_module.flux_pipe,
                        prompt=prompt,
                        conditions=[condition],
                        height=target_size[1],
                        width=target_size[0],
                        generator=generator,
                        model_config=pl_module.model_config,
                        kv_cache=pl_module.model_config.get("independent_condition", False)
                    )

                    generated_image = res.images[0]
                    generated_images.append(generated_image)

                for idx, generated_image in enumerate(generated_images):
                    generated_image.save(save_dir / f"samples/{global_idx+idx}.jpg")

                controls = torch.clamp(einops.rearrange(batch['condition_0'], 'b c h w -> b h w c')*255, 0., 255.).cpu().squeeze(-1).numpy().astype(np.uint8)
                gt_detections = batch['gt']

                for idx, control in enumerate(controls):
                    Image.fromarray(control).save(save_dir / f"controls/{global_idx+idx}.png")
                    with open(save_dir / f"controls/{global_idx+idx}.json", "w") as f:
                        json.dump(gt_detections[idx], f, indent=4)

                global_idx += bs

                batch_results = self.evaluator(generated_images, gt_detections)
                for k in batch_results.keys():
                    results[k].extend(batch_results[k])
            for k in results.keys():
                pl_module.logger.experiment.add_scalar(k, np.nanmean(results[k]), global_step)

        print("Evaluation completed. Moving on to training ...")

    # @rank_zero
    def launch_eval(self, model, global_step):
        self.eval(model, global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        is_last_batch_in_accumulation = (batch_idx + 1) % trainer.accumulate_grad_batches == 0
        if is_last_batch_in_accumulation and trainer.global_step % self.batch_freq == 0:
            self.launch_eval(pl_module, trainer.global_step)