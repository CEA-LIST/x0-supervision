from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils.logger import TensorBoardImageLogger
from cldm.model import create_model, load_state_dict, create_dataloader
from pytorch_lightning import seed_everything
from pathlib import Path
from argparse import ArgumentParser
import traceback
import warnings

## CLI arguments
parser = ArgumentParser()
parser.add_argument('--config', type=str, help='Configuration file path.')
parser.add_argument('--experiment_label', type=str, help='Label of the experiment.')
parser.add_argument('--experiment_version', type=str, help='Version of the experiment.')
parser.add_argument('--pretrained_init_path', type=str, default=None, help='Weights path for initializing the model.')
parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help="Directory where to save the model checkpoints.")
parser.add_argument('--resume_path', type=str, default=None, help='Model checkpoint to resume training from.')
parser.add_argument('--sd_locked', type=bool, default=True, help='Whether to train the UNet decoder along with the ControlNet.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency.')
parser.add_argument('--max_steps', type=int, default=3000, help='Maximum number of training steps.')
parser.add_argument('--cfg_scale', type=float, default=7.5, help="Classifier-free guidance scale.")
parser.add_argument('--precision', type=str, default='32', help='The training precision')
parser.add_argument('--grad_accumulation_steps', type=int, default=1, help="Gradient accumulation steps.")
parser.add_argument('--seed', type=int, default=None, help='Seed for reproducability.')
args = parser.parse_args()
##


if __name__ == '__main__':
    if args.seed:
        print(f"Setting seed to {args.seed}")
        seed_everything(args.seed)


    train_dataloader, val_dataloader = create_dataloader(args.config)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    print(f"Creating the model")
    model = create_model(args.config).cpu()

    if args.pretrained_init_path is not None:
        warnings.warn('If "pretrained_init_path" is specified, then a new training will be started even if "resume_path" is given.')
        print("\nLoading model pretrained weights for initializing the model ...\n")
        missing, unexpected = model.load_state_dict(load_state_dict(args.pretrained_init_path, location='cpu'), strict=False)
        print(f"Restored from {args.pretrained_init_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")



    model.sd_locked = args.sd_locked
    model.learning_rate = args.lr
    model.only_mid_control = False
    

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='_'.join([args.experiment_label, args.experiment_version]) + '_{epoch:02d}-{step}',
        save_weights_only=False,
        every_n_train_steps=args.log_freq,
        save_top_k=1,  # Only save the latest checkpoint
    )

    tensorboard_logger = TensorBoardLogger(save_dir="lightning_logs", name=args.experiment_label, version=args.experiment_version)

    image_logger = TensorBoardImageLogger(
        batch_frequency=args.log_freq,
        log_images_kwargs=dict(
            inpaint=False,
            quantize_denoised=False,
            plot_diffusion_rows=True,
            plot_progressive_rows=True,
            unconditional_guidance_scale=args.cfg_scale
        )
    )

    print("\nInitializing the trainer...\n")

    precision = args.precision
    if precision in ['64', '32', '16']:
        precision = int(precision)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        precision=precision,
        logger=tensorboard_logger,
        callbacks=[image_logger, checkpoint_callback],
        accumulate_grad_batches=args.grad_accumulation_steps,
        max_steps=args.max_steps,
        log_every_n_steps=args.log_freq,
        benchmark=True
    )

    # Train!
    try:
        resume_training = args.resume_path is not None and args.pretrained_init_path is not None
        print("\nTraining start...\n")
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=(args.resume_path if resume_training else None))
        print("Training end")
    except:
        save_dir = Path(args.ckpt_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        save_path = save_dir / f'{args.experiment_label}-{args.experiment_version}-last.ckpt'
        trainer.save_checkpoint(save_path)

        traceback.print_exc()