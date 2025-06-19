#!/usr/bin/env python3
"""
Training script for fine-tuning optical flow models on noisy environments.

This script extends the standard ptlflow training to incorporate various noise models
(starlight, eld) during training to improve robustness in low-light and noisy conditions.

Usage:
    python train_noisy.py --model raft --data.train_dataset flying_chairs --noise_model starlight
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from jsonargparse import ArgumentParser
from loguru import logger

# Add parent directory to path to import noise module
sys.path.append(str(Path(__file__).parent.parent))
from noise import generate_noise, reshape_noise_params, actual_labels_starlight, actual_labels_eld

from ptlflow.data.flow_datamodule import FlowDataModule
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.lightning.ptlflow_trainer import PTLFlowTrainer
from ptlflow.utils.registry import RegisteredModel
from ptlflow.models.base_model.base_model import BaseModel


class NoisyFlowDataModule(FlowDataModule):
    """Extended FlowDataModule that applies noise during training."""
    
    def __init__(
        self,
        noise_model: str = "starlight",
        noise_probability: float = 0.5,
        noise_params_range: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.noise_model = noise_model
        self.noise_probability = noise_probability
        
        # Default noise parameter ranges (0-1, will be scaled in generate_noise)
        if noise_params_range is None:
            if noise_model == "starlight":
                self.noise_params_range = {
                    'alpha_brightness': [0.2, 0.8],
                    'gamma_brightness': [0.2, 0.8], 
                    'shot_noise': [0.0, 0.8],
                    'read_noise': [0.0, 0.6],
                    'quant_noise': [0.0, 0.6],
                    'band_noise': [0.0, 0.4],
                    'band_noise_temp': [0.0, 0.4],
                    'periodic0': [0.0, 0.6],
                    'periodic1': [0.0, 0.6],
                    'periodic2': [0.0, 0.6],
                    'band_noise_angle': [0.0, 1.0]  # 0 or 1
                }
            else:  # eld
                self.noise_params_range = {
                    'alpha_brightness': [0.2, 0.8],
                    'gamma_brightness': [0.2, 0.8],
                    'shot_noise_log': [0.0, 0.8],
                    'read_noise_scale': [0.0, 0.6],
                    'read_noise_tlambda': [0.0, 0.8],
                    'quant_noise': [0.0, 0.6],
                    'band_noise': [0.0, 0.4],
                    'band_noise_angle': [0.0, 1.0]
                }
        else:
            self.noise_params_range = noise_params_range

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Override to apply noise after moving to device."""
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        
        # Only apply noise during training
        if self.trainer and self.trainer.training:
            batch = self._apply_noise_to_batch(batch)
        
        return batch

    def _apply_noise_to_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply noise to images in the batch."""
        if torch.rand(1).item() > self.noise_probability:
            return batch
            
        # Get images from batch
        images = batch.get("images", None)
        if images is None:
            return batch
            
        # Handle different image tensor shapes
        original_shape = images.shape
        if len(original_shape) == 5:  # B, N, C, H, W
            B, N, C, H, W = original_shape
        elif len(original_shape) == 4:  # B, C, H, W (single image pair case)
            B, C, H, W = original_shape
            N = 1
            images = images.unsqueeze(1)  # Add frame dimension
        else:
            logger.warning(f"Unexpected image shape: {original_shape}")
            return batch
            
        device = images.device
        
        # Generate random noise parameters for each item in batch
        noise_params_list = []
        param_names = list(self.noise_params_range.keys())
        
        for b in range(B):
            noise_params_item = []
            for param_name in param_names:
                min_val, max_val = self.noise_params_range[param_name]
                if param_name == 'band_noise_angle':
                    # Binary choice for banding angle
                    param_val = torch.randint(0, 2, (1,)).float().item()
                else:
                    param_val = torch.rand(1).item() * (max_val - min_val) + min_val
                noise_params_item.append(param_val)
            noise_params_list.append(noise_params_item)
        
        noise_params = torch.tensor(noise_params_list, device=device)  # (B, num_params)
        
        # Reshape noise parameters
        noise_dict = reshape_noise_params(noise_params, self.noise_model, num_frames=N)
        
        # Apply noise to images
        images_flat = images.view(B * N, C, H, W)
        
        # Convert to [0, 1] range if needed
        if images_flat.max() > 1.0:
            images_flat = images_flat / 255.0
            was_scaled = True
        else:
            was_scaled = False
            
        # Generate noisy images
        try:
            noisy_images = generate_noise(images_flat, noise_dict, self.noise_model, num_frames=N, device=device)
            
            # Scale back if needed
            if was_scaled:
                noisy_images = noisy_images * 255.0
                
            # Reshape back to original format
            noisy_images = noisy_images.view(B, N, C, H, W)
            
            # If original was 4D, squeeze back to 4D
            if len(original_shape) == 4:
                noisy_images = noisy_images.squeeze(1)
                
            batch["images"] = noisy_images
            
        except Exception as e:
            logger.warning(f"Failed to apply noise: {e}, using original images")
            
        return batch


class NoisyBaseModel(BaseModel):
    """Extended BaseModel that can handle noisy training."""
    
    def training_step(self, batch, batch_idx):
        # The noise application is handled in the datamodule
        return super().training_step(batch, batch_idx)


def _init_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wdecay", type=float, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--project", type=str, default="ptlflow_noisy")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--train_ckpt_topk", type=int, default=0)
    parser.add_argument("--train_ckpt_metric", type=str, default="train/loss_epoch")
    parser.add_argument("--infer_ckpt_topk", type=int, default=1)
    parser.add_argument("--infer_ckpt_metric", type=str, default="main_val_metric")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--log_dir", type=str, default="ptlflow_logs_noisy")
    
    # Noise-specific arguments
    parser.add_argument(
        "--noise_model", 
        type=str, 
        default="starlight",
        choices=["starlight", "eld"],
        help="Noise model to use during training"
    )
    parser.add_argument(
        "--noise_probability", 
        type=float, 
        default=0.5,
        help="Probability of applying noise to each batch"
    )
    parser.add_argument(
        "--output_weights_dir",
        type=str,
        default="weights_noisy",
        help="Directory to save the final trained weights"
    )
    
    return parser


def cli_main():
    parser = _init_parser()

    cfg = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        datamodule_class=NoisyFlowDataModule,
        trainer_class=PTLFlowTrainer,
        auto_configure_optimizers=False,
        parser_kwargs={"parents": [parser]},
        run=False,
        parse_only=True,
    ).config

    model_name = cfg.model.class_path.split(".")[-1]

    # Add noise parameters to datamodule config
    cfg.data.init_args.noise_model = cfg.noise_model
    cfg.data.init_args.noise_probability = cfg.noise_probability

    # Setup loggers and callbacks
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    log_model_name = (
        f"{model_name}-{_gen_dataset_id(cfg.data.train_dataset)}-{cfg.noise_model}-{timestamp}"
    )
    log_model_dir = Path(cfg.log_dir) / log_model_name
    log_model_dir.mkdir(parents=True, exist_ok=True)
    
    if cfg.logger == "tensorboard":
        trainer_logger = {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {
                "save_dir": str(log_model_dir),
                "name": f"{model_name}-{cfg.data.train_dataset}-{cfg.noise_model}",
                "version": cfg.version,
            },
        }
    elif cfg.logger == "wandb":
        trainer_logger = {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            "init_args": {
                "save_dir": str(log_model_dir),
                "project": cfg.project,
                "version": cfg.version,
                "name": f"{model_name}-{cfg.data.train_dataset}-{cfg.noise_model}",
            },
        }

    callbacks = []

    # Model checkpoints
    callbacks.append(
        {
            "class_path": "lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint",
            "init_args": {
                "filename": f"{model_name}_{cfg.noise_model}_last_{{epoch}}_{{step}}",
                "save_weights_only": True,
                "mode": "max",
            },
        }
    )

    callbacks.append(
        {
            "class_path": "lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint",
            "init_args": {
                "filename": f"{model_name}_{cfg.noise_model}_train_{{epoch}}_{{step}}",
            },
        }
    )

    if cfg.train_ckpt_topk > 0:
        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint",
                "init_args": {
                    "filename": f"{model_name}_{cfg.noise_model}"
                    + "_{"
                    + cfg.train_ckpt_metric
                    + ":.2f}_{epoch}",
                    "save_weights_only": False,
                    "save_top_k": cfg.train_ckpt_topk,
                    "monitor": "train/loss_epoch",
                },
            }
        )

    if cfg.infer_ckpt_topk > 0:
        assert (
            cfg.infer_ckpt_metric is not None
        ), "You must provide a metric name for --infer_ckpt_topk_metric"
        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint",
                "init_args": {
                    "filename": f"{model_name}_{cfg.noise_model}_best"
                    + "_{"
                    + cfg.infer_ckpt_metric
                    + ":.2f}_{epoch}_{step}",
                    "save_weights_only": True,
                    "save_top_k": cfg.infer_ckpt_topk,
                    "monitor": cfg.infer_ckpt_metric,
                    "dirpath": cfg.output_weights_dir,
                },
            }
        )

    callbacks.append(
        {
            "class_path": "ptlflow.utils.callbacks.logger.LoggerCallback",
        }
    )

    cfg.trainer.logger = trainer_logger
    cfg.trainer.callbacks = callbacks
    cfg.model.init_args.lr = cfg.lr
    cfg.model.init_args.wdecay = cfg.wdecay
    
    cli = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        trainer_class=PTLFlowTrainer,
        datamodule_class=NoisyFlowDataModule,
        auto_configure_optimizers=False,
        args=cfg,
        run=False,
        ignore_sys_argv=True,
        parser_kwargs={"parents": [parser]},
    )

    logger.info(f"Training {model_name} with {cfg.noise_model} noise model")
    logger.info(f"Noise probability: {cfg.noise_probability}")
    logger.info(f"Output weights will be saved to: {cfg.output_weights_dir}")

    if not cli.model.has_trained_on_ptlflow:
        _print_untested_warning()

    # Create output weights directory
    os.makedirs(cfg.output_weights_dir, exist_ok=True)

    # Train the model
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=cfg.ckpt_path)

    # Save the final model weights in a format compatible with infer.py
    final_weights_path = Path(cfg.output_weights_dir) / f"{model_name}_{cfg.noise_model}_final.pth"
    torch.save(cli.model.state_dict(), final_weights_path)
    logger.info(f"Final weights saved to: {final_weights_path}")

    # Save training configuration
    config_path = Path(cfg.output_weights_dir) / f"{model_name}_{cfg.noise_model}_config.txt"
    with open(config_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Noise model: {cfg.noise_model}\n")
        f.write(f"Noise probability: {cfg.noise_probability}\n")
        f.write(f"Dataset: {cfg.data.train_dataset}\n")
        f.write(f"Training timestamp: {timestamp}\n")
        f.write(f"Final weights: {final_weights_path}\n")


def _gen_dataset_id(dataset_string: str) -> str:
    sep_datasets = dataset_string.split("+")
    names_list = []
    for dataset in sep_datasets:
        if "*" in dataset:
            tokens = dataset.split("*")
            try:
                _, dataset_params = int(tokens[0]), tokens[1]
            except ValueError:  # the multiplier is at the end
                dataset_params = tokens[0]
        else:
            dataset_params = dataset

        dataset_name = dataset_params.split("-")[0]
        names_list.append(dataset_name)

    dataset_id = "_".join(names_list)
    return dataset_id


def _print_untested_warning():
    print("###########################################################################")
    print("# WARNING: Noisy training experimental feature!                          #")
    print("#                                                                         #")
    print("# This is an experimental script for training with synthetic noise.      #")
    print("# The noise models are applied during training to improve robustness     #")
    print("# in low-light and noisy environments.                                   #")
    print("#                                                                         #")
    print("# Supported noise models: starlight, eld                                 #")
    print("###########################################################################")


def _show_v04_warning():
    ignore_args = ["-h", "--help", "--model", "--config"]
    for arg in ignore_args:
        if arg in sys.argv:
            return

    logger.warning(
        "Since v0.4, it is now necessary to inform the model using the --model argument. For example, use: python train_noisy.py --model raft --data.train_dataset flying_chairs --noise_model starlight"
    )


if __name__ == "__main__":
    _show_v04_warning()
    cli_main() 