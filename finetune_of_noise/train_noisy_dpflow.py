#!/usr/bin/env python3
"""
Finetuning DPFlow models forward flow on synthetic noisy applied to sintel dataset
"""

import os
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import types
from typing import Dict, Any, Optional

from loguru import logger
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torchvision.transforms.functional import equalize, gaussian_blur

from noise import generate_noise, reshape_noise_params
from ptlflow.data.flow_datamodule import FlowDataModule
from ptlflow.utils.lightning.ptlflow_trainer import PTLFlowTrainer
import ptlflow


class PrintLossCallback(Callback):
    """A callback to print the training loss at each step."""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if 'loss' in outputs:
            loss = outputs['loss'].item()
            logger.info(f"Epoch {trainer.current_epoch}, Step {trainer.global_step}: train_loss = {loss:.4f}")


class SaveWeightsOnlyCallback(Callback):
    """A callback to save only the model weights at the end of training."""
    def __init__(self, training_args: Dict[str, Any]):
        super().__init__()
        self.training_args = training_args

    def on_train_end(self, trainer, pl_module):
        output_dir = "finetune_of_noise/weights"
        os.makedirs(output_dir, exist_ok=True)
        model_name = self.training_args["model"]
        dataset_name = self.training_args["train_dataset"]
        output_filename = f"{model_name}-{dataset_name}-enhancement-finetuned.pth"
        output_path = os.path.join(output_dir, output_filename)
        logger.info(f"Training finished. Saving final model weights to {output_path}")
        state_dict = pl_module.state_dict()
        torch.save(state_dict, output_path)
        logger.info(f"Successfully saved weights to {output_path}")


class NoisyFlowDataModule(FlowDataModule):
    """Extended FlowDataModule that applies noise during training."""
    def __init__(
        self,
        noise_model: str = "starlight",
        noise_probability: float = 0.8, 
        noise_params_range: Optional[Dict[str, float]] = None,
        val_crop_size: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if val_crop_size:
            self.val_crop_size = val_crop_size
        self.noise_model = noise_model
        self.noise_probability = noise_probability
        if noise_params_range is None:
            if noise_model == "starlight":
                self.noise_params_range = {
                    "alpha_brightness": [0.05, 0.4], "gamma_brightness": [0.05, 0.4],
                    "shot_noise": [0.4, 1.5], "read_noise": [0.5, 1.2],
                    "quant_noise": [0.2, 1.0], "band_noise": [0.2, 0.8],
                    "band_noise_temp": [0.2, 0.8], "periodic0": [0.0, 0.6],
                    "periodic1": [0.0, 0.6], "periodic2": [0.0, 0.6],
                    "band_noise_angle": [0.0, 1.0],
                }
            else:  
                self.noise_params_range = {
                    "alpha_brightness": [0.1, 0.6], "gamma_brightness": [0.1, 0.6],
                    "shot_noise_log": [0.3, 1.0], "read_noise_scale": [0.3, 0.8],
                    "read_noise_tlambda": [0.2, 0.9], "quant_noise": [0.1, 0.7],
                    "band_noise": [0.1, 0.5], "band_noise_angle": [0.0, 1.0],
                }
        else:
            self.noise_params_range = noise_params_range

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if self.trainer and self.trainer.training:
            batch = self._apply_noise_to_batch(batch)
        batch = self._ensure_divisible(batch, divisor=8)
        return batch

    def _ensure_divisible(self, batch: Dict[str, torch.Tensor], divisor: int = 8) -> Dict[str, torch.Tensor]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 4:
                H, W = value.shape[-2:]
                new_H, new_W = (H // divisor) * divisor, (W // divisor) * divisor
                if new_H != H or new_W != W:
                    batch[key] = value[..., :new_H, :new_W]
        return batch

    def _apply_noise_to_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        images = batch.get("images", None)
        if images is None: return batch

        if images.dim() == 4: images = images.unsqueeze(1)
        B, N, C, H, W = images.shape
        assert N >= 2, "Need at least two frames for flow estimation"

        prev_frame_clean = images[:, 0]
        curr_frame_clean = images[:, 1]
        device = images.device

        # Create the "pseudo H3" from the PREVIOUS frame
        illumination1 = gaussian_blur(prev_frame_clean, kernel_size=21, sigma=10)
        pseudo_h3 = (prev_frame_clean / (illumination1 + 1e-6)).clamp(0, 1.0) * 255.0

        # Create the "pseudo L2" from the CURRENT frame
        noisy_img2 = curr_frame_clean
        if torch.rand(1).item() <= self.noise_probability:
            rows = [[torch.rand(1).item() for _ in self.noise_params_range] for _ in range(B)]
            noise_tensor = torch.tensor(rows, device=device)
            noise_dict = reshape_noise_params(noise_tensor, self.noise_model, num_frames=1)

            scaled = curr_frame_clean.max() > 1.0
            curr_frame_norm = curr_frame_clean / 255.0 if scaled else curr_frame_clean
            noisy_img2_norm = generate_noise(curr_frame_norm, noise_dict, self.noise_model, num_frames=1, device=device)
            noisy_img2 = noisy_img2_norm * 255.0 if scaled else noisy_img2_norm

        # Gaussian blur is used to simulate the effect of the main model's Denoise_1 kernel.
        pseudo_l2_denoised = gaussian_blur(noisy_img2, kernel_size=5, sigma=1.5)
        
        pseudo_l2_equalized = torch.zeros_like(pseudo_l2_denoised)
        pseudo_h3_equalized = torch.zeros_like(pseudo_h3)
        for b in range(B):
            l2_uint8 = pseudo_l2_denoised[b].clamp(0, 255).to(torch.uint8)
            h3_uint8 = pseudo_h3[b].clamp(0, 255).to(torch.uint8)
            
            pseudo_l2_equalized[b] = equalize(l2_uint8).float()
            pseudo_h3_equalized[b] = equalize(h3_uint8).float()
            
        # Stack the images in the correct (L2 -> H3) order
        new_imgs = torch.stack([pseudo_h3_equalized, pseudo_l2_equalized], dim=1)
        
        original_shape = images.shape
        if len(original_shape) == 4:
            new_imgs = new_imgs.squeeze(1)
        batch["images"] = new_imgs
        return batch


def main():
    log_dir = "finetune_of_noise/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"train_noisy_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    logger.add(log_file_path, rotation="10 MB")

    args = {
        "model": "dpflow",
        "ckpt_path": "things",
        "train_dataset": "sintel",
        "val_dataset": "sintel",
        "mpi_sintel_root_dir": "./finetune_of_noise/MPI-Sintel-complete/",
        "noise_model": "starlight",
        "noise_probability": 0.8,
        "train_batch_size": 4,
        "lr": 5e-5,
        "max_epochs": 100,
        "accelerator": "auto",
        "sintel_dstype": "final",
        "gradient_clip_val": 1.0,
        "val_check_interval": 0.25,
        "train_crop_size": [320, 640],
        "val_crop_size": [320, 640],
        "corr_mode": "allpairs",
    }
    _print_untested_warning()

    logger.info(f"Loading model: {args['model']}")
    model = ptlflow.get_model(args["model"], ckpt_path=args["ckpt_path"])
    model.lr = args["lr"]

    model.output_stride = 8
    
    if hasattr(model, 'pyramid_levels'):
        model.pyramid_levels = 3
    if hasattr(model, 'mixed_precision'):
        model.mixed_precision = True

    try:
        from ptlflow.models.dpflow.update import ConvGRU as _DPConvGRU
        from ptlflow.models.dpflow.update import CGUGRU as _DPCGUGRU
        def _safe_cat_forward(self, h, x):
            if h.shape[-2:] != x.shape[-2:]:
                _, _, Hh, Wh = h.shape
                _, _, Hx, Wx = x.shape
                min_h, min_w = min(Hh, Hx), min(Wh, Wx)
                h = h[..., :min_h, :min_w]
                x = x[..., :min_h, :min_w]
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(self.convz(hx)) if hasattr(self, 'convz') else None
            r = torch.sigmoid(self.convr(hx))
            q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
            h = (1 - z) * h + z * q if z is not None else h
            return h
        _DPConvGRU.forward = _safe_cat_forward
        _DPCGUGRU.forward = _safe_cat_forward
    except Exception as e:
        logger.warning(f"Failed to patch ConvGRU for size mismatches: {e}")

    logger.info(f"Loading dataset: {args['train_dataset']}")
    datamodule = NoisyFlowDataModule(
        train_dataset=args["train_dataset"],
        val_dataset=args["val_dataset"],
        train_batch_size=args["train_batch_size"],
        noise_model=args["noise_model"],
        noise_probability=args["noise_probability"],
        train_crop_size=args["train_crop_size"],
        val_crop_size=args["val_crop_size"],
    )

    datamodule.sintel_root_dir = args["mpi_sintel_root_dir"]
    datamodule.mpi_sintel_root_dir = args["mpi_sintel_root_dir"]
    datamodule.sintel_dstype = args["sintel_dstype"]
    
    def _patched_load_dataset_paths(self):
        self.dataset_paths = {'sintel': self.sintel_root_dir}
    datamodule._load_dataset_paths = types.MethodType(_patched_load_dataset_paths, datamodule)

    logger.info("Setting up trainer.")
    early_stopping_callback = EarlyStopping(monitor="val_sintel_clean_final/val/epe", patience=5, mode="min")

    model.optimizer = torch.optim.AdamW
    model.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    trainer = PTLFlowTrainer(
        accelerator=args["accelerator"], max_epochs=args["max_epochs"],
        gradient_clip_val=args["gradient_clip_val"], val_check_interval=args["val_check_interval"],
        num_sanity_val_steps=0,
        callbacks=[PrintLossCallback(), SaveWeightsOnlyCallback(args), early_stopping_callback],
        enable_checkpointing=False,
    )

    logger.info("Starting training.")
    trainer.fit(model, datamodule)
    logger.info("Training completed successfully!")


def _print_untested_warning():
    print("###########################################################################")
    print("#           Training with ENHANCEMENT-STYLE data simulation             #")
    print("###########################################################################")


if __name__ == "__main__":
    main()