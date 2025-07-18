#!/usr/bin/env python3
"""
Fine-tune RAFT on data that perfectly mimics the enhancement network's inputs.

This script simulates the data format used in the main neural network to 
achieve the best possible performance by closing the domain gap. It generates a 
pseudo-reflectance map (last_H3) and a pseudo-denoised, equalized 
illumination map (L2) for fine-tuning the RAFT model.
"""

import os
from datetime import datetime
import types
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from loguru import logger
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torchvision.transforms.functional import equalize, gaussian_blur

import ptlflow
from noise import generate_noise, reshape_noise_params
from ptlflow.data.flow_datamodule import FlowDataModule
from ptlflow.utils.lightning.ptlflow_trainer import PTLFlowTrainer

# -----------------------------------------------------------------------------
#  Callbacks (Unchanged)
# -----------------------------------------------------------------------------
class PrintLossCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "loss" in outputs:
            logger.info(
                f"Epoch {trainer.current_epoch} – Step {trainer.global_step}:"
                f" loss = {outputs['loss'].item():.4f}"
            )

class SaveWeightsOnlyCallback(Callback):
    def __init__(self, training_args: Dict[str, Any]):
        super().__init__()
        self.training_args = training_args

    def on_train_end(self, trainer, pl_module):
        out_dir = "finetune_of_noise/weights"
        os.makedirs(out_dir, exist_ok=True)
        fname = f"raft-{self.training_args['train_dataset']}-finetuned-fix-1.pth"
        path = os.path.join(out_dir, fname)
        torch.save(pl_module.state_dict(), path)
        logger.info(f"Saved fine-tuned weights to {path}")

# -----------------------------------------------------------------------------
#  DataModule with Enhancement-Style Data Simulation
# -----------------------------------------------------------------------------
class EnhancementFlowDataModule(FlowDataModule):
    def __init__(
        self,
        noise_model: str = "starlight",
        noise_probability: float = 0.8, # Increased probability for more robust training
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
            batch = self._simulate_enhancement_inputs(batch)
        batch = self._ensure_divisible(batch, divisor=8)
        return batch

    def _ensure_divisible(self, batch, divisor=8):
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor) or v.dim() < 4: continue
            shape = v.shape
            H, W = shape[-2:]
            new_H, new_W = (H // divisor) * divisor, (W // divisor) * divisor
            if (new_H, new_W) != (H, W):
                batch[k] = v[..., :new_H, :new_W]
        return batch

    def _simulate_enhancement_inputs(self, batch):
        imgs = batch.get("images")
        if imgs is None: return batch

        if imgs.dim() == 4: imgs = imgs.unsqueeze(1)
        B, N, C, H, W = imgs.shape
        assert N >= 2, "Need at least two frames"

        prev_frame_clean = imgs[:, 0]
        curr_frame_clean = imgs[:, 1]
        device = imgs.device

        # --- 1. Generate the "pseudo last_H3" from the PREVIOUS frame ---
        # Rationale: This simulates the texture-rich, illumination-independent
        # reflectance map (H3) from model.py.
        illumination1 = gaussian_blur(prev_frame_clean, kernel_size=21, sigma=10)
        # --- FIX: Scale to [0, 255] as done in model.py's update_cache ---
        pseudo_h3 = (prev_frame_clean / (illumination1 + 1e-6)).clamp(0, 1.0) * 255.0

        # --- 2. Generate the "pseudo L2" from the CURRENT frame ---
        # Rationale: This simulates the initial denoised and then equalized
        # illumination component (L2) from model.py.
        noisy_img2 = curr_frame_clean
        if torch.rand(1).item() <= self.noise_probability:
            rows = []
            names = list(self.noise_params_range.keys())
            for _ in range(B):
                row = [torch.rand(1).item() * (hi - lo) + lo for name, (lo, hi) in self.noise_params_range.items()]
                rows.append(row)
            noise_tensor = torch.tensor(rows, device=device)
            noise_dict = reshape_noise_params(noise_tensor, self.noise_model, num_frames=1)
            
            scaled = curr_frame_clean.max() > 1.0
            curr_frame_norm = curr_frame_clean / 255.0 if scaled else curr_frame_clean
            noisy_img2_norm = generate_noise(curr_frame_norm, noise_dict, self.noise_model, num_frames=1, device=device)
            noisy_img2 = noisy_img2_norm * 255.0 if scaled else noisy_img2_norm

        # Simulate Denoise_1 with a mild blur
        pseudo_l2_denoised = gaussian_blur(noisy_img2, kernel_size=5, sigma=1.5)
        
        # Apply histogram equalization exactly as in update_cache
        pseudo_l2_equalized = torch.zeros_like(pseudo_l2_denoised)
        for b in range(B):
            frame = pseudo_l2_denoised[b]
            frame_uint8 = frame.clamp(0, 255).to(torch.uint8)
            pseudo_l2_equalized[b] = equalize(frame_uint8).float()
            
        # --- 3. Replace batch images with the correctly ordered simulated data ---
        # --- CRITICAL FIX & RATIONALE ---
        # In model.py, RAFT is called as: `self.of_model(L2_tmp, last_H3_tmp, ...)`
        # This means the model must be trained to predict flow FROM the equalized
        # L2 map TO the H3 reflectance map. We stack the tensors in this exact
        # order to match the inference call.
        new_imgs = torch.stack([pseudo_l2_equalized, pseudo_h3], dim=1) # (Current, Previous)
        
        if imgs.dim() == 4: new_imgs = new_imgs.squeeze(1)
        batch["images"] = new_imgs
        return batch

# -----------------------------------------------------------------------------
#  Main (Unchanged)
# -----------------------------------------------------------------------------
def _print_warning():
    print("# Fine-tuning RAFT on ENHANCEMENT-STYLE data – EXPERIMENTAL")

def main():
    _print_warning()
    log_dir = "finetune_of_noise/logs"
    os.makedirs(log_dir, exist_ok=True)
    fname = f"train_enhancement_raft_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(os.path.join(log_dir, fname), rotation="10 MB")

    args = {
        "model": "raft", "ckpt_path": "things", "train_dataset": "sintel",
        "val_dataset": "sintel", "mpi_sintel_root_dir": "./finetune_of_noise/MPI-Sintel-complete/",
        "noise_model": "starlight", "noise_probability": 0.8, "train_batch_size": 4,
        "lr": 2e-5, "max_epochs": 100, "accelerator": "auto", "sintel_dstype": "final",
        "val_check_interval": 0.25, "train_crop_size": [320, 640], "val_crop_size": [320, 640],
    }

    logger.info(f"Loading RAFT model from {args['ckpt_path']}")
    model = ptlflow.get_model(args["model"], ckpt_path=args["ckpt_path"])
    model.lr = args["lr"]

    datamodule = EnhancementFlowDataModule(
        train_dataset=args["train_dataset"], val_dataset=args["val_dataset"],
        train_batch_size=args["train_batch_size"], noise_model=args["noise_model"],
        noise_probability=args["noise_probability"], train_crop_size=args["train_crop_size"],
        val_crop_size=args["val_crop_size"],
    )
    datamodule.sintel_root_dir = args["mpi_sintel_root_dir"]
    datamodule.mpi_sintel_root_dir = args["mpi_sintel_root_dir"]
    datamodule.sintel_dstype = args["sintel_dstype"]
    def _patched(self): self.dataset_paths = {"sintel": self.sintel_root_dir}
    datamodule._load_dataset_paths = types.MethodType(_patched, datamodule)

    early_stop = EarlyStopping(monitor="val_sintel_clean_final/val/epe", patience=5, mode="min")
    trainer = PTLFlowTrainer(
        accelerator=args["accelerator"], max_epochs=args["max_epochs"],
        val_check_interval=args["val_check_interval"], num_sanity_val_steps=0,
        gradient_clip_val=1.0, callbacks=[PrintLossCallback(), SaveWeightsOnlyCallback(args), early_stop],
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule)
    logger.info("RAFT fine-tuning on enhancement data completed!")

if __name__ == "__main__":
    main()