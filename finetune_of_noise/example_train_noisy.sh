#!/bin/bash

# Example script for training optical flow models with noise

# First, make sure your dataset paths are configured in datasets.yaml
# Edit ptlflow_scripts/datasets.yaml to point to your dataset locations

# Example 1: Train RAFT on Flying Chairs with starlight noise
echo "Training RAFT with starlight noise on Flying Chairs..."
python train_noisy.py \
    --model raft \
    --data.train_dataset flying_chairs \
    --data.val_dataset flying_chairs \
    --noise_model starlight \
    --noise_probability 0.5 \
    --output_weights_dir ./weights_noisy/raft_starlight \
    --trainer.max_epochs 50 \
    --trainer.accelerator gpu \
    --trainer.devices 1 \
    --lr 1e-4 \
    --trainer.log_every_n_steps 100

# Example 2: Train with ELD noise model
echo "Training RAFT with ELD noise on Flying Chairs..."
python train_noisy.py \
    --model raft \
    --data.train_dataset flying_chairs \
    --data.val_dataset flying_chairs \
    --noise_model eld \
    --noise_probability 0.7 \
    --output_weights_dir ./weights_noisy/raft_eld \
    --trainer.max_epochs 50 \
    --trainer.accelerator gpu \
    --trainer.devices 1 \
    --lr 1e-4

# Example 3: Fine-tune from pretrained weights
echo "Fine-tuning from pretrained RAFT..."
python train_noisy.py \
    --model raft \
    --ckpt_path raft-things \
    --data.train_dataset sintel \
    --data.val_dataset sintel \
    --noise_model starlight \
    --noise_probability 0.4 \
    --output_weights_dir ./weights_noisy/raft_starlight_finetune \
    --trainer.max_epochs 20 \
    --trainer.accelerator gpu \
    --trainer.devices 1 \
    --lr 5e-5

# Example 4: Training with custom noise parameters
echo "Training with custom noise settings..."
python train_noisy.py \
    --model raft \
    --data.train_dataset flying_chairs \
    --data.val_dataset flying_chairs \
    --noise_model starlight \
    --noise_probability 0.8 \
    --output_weights_dir ./weights_noisy/raft_heavy_noise \
    --trainer.max_epochs 30 \
    --trainer.accelerator gpu \
    --trainer.devices 1 \
    --lr 1e-4

echo "Training completed! Check the weights_noisy directory for saved models."
echo "Use infer.py with the saved weights to test on noisy images." 