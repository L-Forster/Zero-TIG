#!/usr/bin/env python3
"""
Training script for fine-tuning optical flow models on noisy environments.

This script extends the standard ptlflow training to incorporate various noise models
(starlight, eld) during training to improve robustness in low-light and noisy conditions.

Usage:
    This script is now configured programmatically. See the `main` function
    to adjust training parameters.
"""

import os
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import types
from typing import Dict, Any, Optional

from loguru import logger
from lightning.pytorch.callbacks import Callback

from noise import generate_noise, reshape_noise_params
from ptlflow.data.flow_datamodule import FlowDataModule
from ptlflow.utils.lightning.ptlflow_trainer import PTLFlowTrainer
import ptlflow


class PrintLossCallback(Callback):
    """A callback to print the training loss at each step."""

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        """Called when the train batch ends."""
        if 'loss' in outputs:
            loss = outputs['loss'].item()
            logger.info(f"Epoch {trainer.current_epoch}, Step {trainer.global_step}: train_loss = {loss:.4f}")


class SaveWeightsOnlyCallback(Callback):
    """A callback to save only the model weights at the end of training."""
    def on_train_end(self, trainer, pl_module):
        """Called when the train ends."""
        output_dir = "finetune_of_noise/weights"
        os.makedirs(output_dir, exist_ok=True)
        
        model_name = trainer.model.hparams.model
        dataset_name = trainer.datamodule.hparams.train_dataset
        output_filename = f"{model_name}-{dataset_name}-noisy.pth"
        output_path = os.path.join(output_dir, output_filename)

        logger.info(f"Training finished. Saving final model weights to {output_path}")

        state_dict = pl_module.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k.replace('model.', '', 1)] = v
            else:
                new_state_dict[k] = v
        
        torch.save(new_state_dict, output_path)
        logger.info(f"Successfully saved weights to {output_path}")


class NoisyFlowDataModule(FlowDataModule):
    """Extended FlowDataModule that applies noise during training."""

    def __init__(
        self,
        noise_model: str = "starlight",
        noise_probability: float = 0.5,
        noise_params_range: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise_model = noise_model
        self.noise_probability = noise_probability

        if noise_params_range is None:
            if noise_model == "starlight":
                self.noise_params_range = {
                    "alpha_brightness": [0.2, 0.8],
                    "gamma_brightness": [0.2, 0.8],
                    "shot_noise": [0.0, 0.8],
                    "read_noise": [0.0, 0.6],
                    "quant_noise": [0.0, 0.6],
                    "band_noise": [0.0, 0.4],
                    "band_noise_temp": [0.0, 0.4],
                    "periodic0": [0.0, 0.6],
                    "periodic1": [0.0, 0.6],
                    "periodic2": [0.0, 0.6],
                    "band_noise_angle": [0.0, 1.0],
                }
            else:  # eld
                self.noise_params_range = {
                    "alpha_brightness": [0.2, 0.8],
                    "gamma_brightness": [0.2, 0.8],
                    "shot_noise_log": [0.0, 0.8],
                    "read_noise_scale": [0.0, 0.6],
                    "read_noise_tlambda": [0.0, 0.8],
                    "quant_noise": [0.0, 0.6],
                    "band_noise": [0.0, 0.4],
                    "band_noise_angle": [0.0, 1.0],
                }
        else:
            self.noise_params_range = noise_params_range

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if self.trainer and self.trainer.training:
            batch = self._apply_noise_to_batch(batch)
        return batch

    def _apply_noise_to_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if torch.rand(1).item() > self.noise_probability:
            return batch

        images = batch.get("images", None)
        if images is None:
            return batch

        original_shape = images.shape
        if len(original_shape) == 5:
            B, N, C, H, W = original_shape
        elif len(original_shape) == 4:
            B, C, H, W = original_shape
            N = 1
            images = images.unsqueeze(1)
        else:
            logger.warning(f"Unexpected image shape: {original_shape}")
            return batch

        device = images.device
        noise_params_list = []
        param_names = list(self.noise_params_range.keys())

        for _ in range(B):
            noise_params_item = []
            for param_name in param_names:
                min_val, max_val = self.noise_params_range[param_name]
                if param_name == "band_noise_angle":
                    param_val = torch.randint(0, 2, (1,)).float().item()
                else:
                    param_val = torch.rand(1).item() * (max_val - min_val) + min_val
                noise_params_item.append(param_val)
            noise_params_list.append(noise_params_item)

        noise_params = torch.tensor(noise_params_list, device=device)
        noise_dict = reshape_noise_params(noise_params, self.noise_model, num_frames=N)
        images_flat = images.view(B * N, C, H, W)

        was_scaled = False
        if images_flat.max() > 1.0:
            images_flat = images_flat / 255.0
            was_scaled = True

        try:
            noisy_images = generate_noise(
                images_flat, noise_dict, self.noise_model, num_frames=N, device=device
            )
            if was_scaled:
                noisy_images = noisy_images * 255.0

            noisy_images = noisy_images.view(B, N, C, H, W)
            if len(original_shape) == 4:
                noisy_images = noisy_images.squeeze(1)
            batch["images"] = noisy_images
        except Exception as e:
            logger.warning(f"Failed to apply noise: {e}, using original images")

        return batch


def main():
    """Main function to run the noisy fine-tuning."""
    # --- 0. Logging Setup ---
    log_dir = "finetune_of_noise/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"train_noisy_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    logger.add(log_file_path, rotation="10 MB")
    logger.info(f"Logs for this run will be saved to {log_file_path}")

    # --- 1. Configuration ---
    args = {
        "model": "dpflow",
        "ckpt_path": "things",
        "train_dataset": "sintel",
        "val_dataset": "sintel",
        "mpi_sintel_root_dir": "./finetune_of_noise/MPI-Sintel-complete/",
        "noise_model": "starlight",
        "noise_probability": 0.5,
        "train_batch_size": 1,
        "lr": 1e-5,
        "max_epochs": 20,
        "accelerator": "auto",
        "sintel_dstype": "final",
    }
    _print_untested_warning()

    # --- 2. Model ---
    logger.info(f"Loading model: {args['model']}")
    model = ptlflow.get_model(args["model"], ckpt_path=args["ckpt_path"])
    model.lr = args["lr"]

    # This is a critical fix for a bug in ptlflow where the model's output
    # stride is not available during datamodule setup, causing a crash.
    model.output_stride = 8

    # --- 3. DataModule ---
    logger.info(f"Loading dataset: {args['train_dataset']}")
    datamodule = NoisyFlowDataModule(
        train_dataset=args["train_dataset"],
        val_dataset=args["val_dataset"],
        train_batch_size=args["train_batch_size"],
        noise_model=args["noise_model"],
        noise_probability=args["noise_probability"],
    )

    # This is the critical fix for the two main bugs in ptlflow:
    # a) It uses both 'sintel_root_dir' and 'mpi_sintel_root_dir'. We fix the mismatch.
    datamodule.sintel_root_dir = args["mpi_sintel_root_dir"]
    datamodule.mpi_sintel_root_dir = args["mpi_sintel_root_dir"]
    datamodule.sintel_dstype = args["sintel_dstype"]
    
    # b) It tries to load a non-existent 'datasets.yml'. We prevent this
    #    by directly monkey-patching the problematic method on the instance.
    def _patched_load_dataset_paths(self):
        """
        This method replaces the faulty one from the parent class.
        The original crashes if datasets.yml is missing. This patch
        correctly populates the internal 'dataset_paths' dictionary
        that the rest of the library depends on.
        """
        logger.info("--- Applying monkey-patch to populate dataset_paths ---")
        self.dataset_paths = {'sintel': self.sintel_root_dir}
    
    datamodule._load_dataset_paths = types.MethodType(_patched_load_dataset_paths, datamodule)


    # --- 4. Trainer ---
    logger.info("Setting up trainer.")
    trainer = PTLFlowTrainer(
        accelerator=args["accelerator"],
        max_epochs=args["max_epochs"],
        callbacks=[PrintLossCallback(), SaveWeightsOnlyCallback()],
        enable_checkpointing=False,
    )

    # --- 5. Run Training ---
    logger.info("Starting training.")
    trainer.fit(model, datamodule)
    logger.info("Training completed successfully!")


def _print_untested_warning():
    """Print a warning about the experimental nature of the script."""
    print("###########################################################################")
    print("# WARNING: Noisy training experimental feature!                          #")
    print("#                                                                         #")
    print("# This is an experimental script for training with synthetic noise.      #")
    print("# The noise models are applied during training to improve robustness     #")
    print("# in low-light and noisy environments.                                   #")
    print("#                                                                         #")
    print("# Supported noise models: starlight, eld                                 #")
    print("###########################################################################")


if __name__ == "__main__":
    main() 