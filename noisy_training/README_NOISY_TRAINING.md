# Noisy Environment Optical Flow Training

This system allows you to fine-tune optical flow models for robust performance in noisy environments using synthetic noise models from `noise.py`.

## Overview

The `train_noisy.py` script extends the standard PTLFlow training pipeline to incorporate various noise models during training, improving model robustness in challenging conditions like low-light scenarios.

### Supported Noise Models

1. **Starlight Noise**: Comprehensive noise model including:
   - Shot noise
   - Read noise
   - Quantization noise
   - Banding noise (horizontal/vertical)
   - Temporal banding noise
   - Periodic noise

2. **ELD Noise**: Extreme low-light denoising noise model including:
   - Shot noise (log-normal)
   - Tukey-lambda read noise
   - Quantization noise
   - Banding noise

## Setup

### 1. Configure Dataset Paths

Edit `datasets.yaml` to point to your dataset locations:

```yaml
flying_chairs: /path/to/FlyingChairs_release
flying_things3d: /path/to/FlyingThings3D
mpi_sintel: /path/to/MPI-Sintel
kitti_2015: /path/to/KITTI/2015
# ... add other datasets
```

### 2. Install Dependencies

Make sure you have PTLFlow and all dependencies installed:

```bash
pip install ptlflow
# Other dependencies as needed
```

## Usage

### Basic Training Command

```bash
cd noisy_training
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
    --lr 1e-4
```

### Key Parameters

- `--model`: Choose optical flow model (raft, gma, flowformer, etc.)
- `--noise_model`: Choose noise model (`starlight` or `eld`)
- `--noise_probability`: Probability of applying noise to each batch (0.0-1.0)
- `--output_weights_dir`: Directory to save trained weights
- `--ckpt_path`: Path to pretrained checkpoint for fine-tuning

### Training Examples

#### 1. Train from Scratch with Starlight Noise

```bash
python train_noisy.py \
    --model raft \
    --data.train_dataset flying_chairs \
    --data.val_dataset flying_chairs \
    --noise_model starlight \
    --noise_probability 0.5 \
    --output_weights_dir ./weights_noisy/raft_starlight \
    --trainer.max_epochs 50
```

#### 2. Fine-tune Pretrained Model with ELD Noise

```bash
python train_noisy.py \
    --model raft \
    --ckpt_path raft-things \
    --data.train_dataset sintel \
    --data.val_dataset sintel \
    --noise_model eld \
    --noise_probability 0.4 \
    --output_weights_dir ./weights_noisy/raft_eld_finetune \
    --trainer.max_epochs 20 \
    --lr 5e-5
```

#### 3. Heavy Noise Training

```bash
python train_noisy.py \
    --model raft \
    --data.train_dataset flying_chairs+flying_things3d \
    --data.val_dataset sintel \
    --noise_model starlight \
    --noise_probability 0.8 \
    --output_weights_dir ./weights_noisy/raft_heavy_noise \
    --trainer.max_epochs 30
```

### Running Example Scripts

Use the provided example script:

```bash
chmod +x example_train_noisy.sh
./example_train_noisy.sh
```

## Using Trained Models

### With infer.py

After training, use the saved weights with the standard inference script:

```bash
python infer.py \
    --model raft \
    --ckpt_path ./weights_noisy/raft_starlight/raft_starlight_final.pth \
    --input_path /path/to/your/noisy/images \
    --output_path ./results
```

### Model Compatibility

The trained models are fully compatible with all PTLFlow inference tools:
- `infer.py` - Single inference
- `validate.py` - Validation on datasets
- `test.py` - Test on benchmark datasets

## Noise Model Details

### Starlight Noise Parameters

The starlight model simulates various sensor noise types:

- **Brightness adjustment**: Alpha and gamma correction
- **Shot noise**: Signal-dependent noise
- **Read noise**: Signal-independent sensor noise
- **Quantization noise**: ADC quantization effects
- **Banding noise**: Sensor line artifacts
- **Periodic noise**: Structured interference patterns

### ELD Noise Parameters

The ELD model focuses on extreme low-light conditions:

- **Shot noise**: Modeled with log-normal distribution
- **Read noise**: Tukey-lambda distribution
- **Quantization effects**: Similar to starlight
- **Banding artifacts**: Horizontal/vertical patterns

## Advanced Configuration

### Custom Noise Parameters

You can modify noise parameter ranges by editing the `NoisyFlowDataModule` class:

```python
# In train_noisy.py, modify the noise_params_range dictionary
self.noise_params_range = {
    'alpha_brightness': [0.1, 0.9],  # Custom range
    'shot_noise': [0.0, 1.0],        # Higher noise levels
    # ... other parameters
}
```

### Training Strategy Tips

1. **Progressive Training**: Start with low noise probability, gradually increase
2. **Mixed Datasets**: Combine multiple datasets for better generalization
3. **Learning Rate**: Use lower learning rates when fine-tuning from pretrained models
4. **Validation**: Use clean validation sets to monitor actual performance

## Output Files

After training, the following files are created:

- `{model}_{noise_model}_final.pth`: Final model weights
- `{model}_{noise_model}_config.txt`: Training configuration
- Various checkpoint files during training
- TensorBoard/WandB logs

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Ensure proper GPU utilization and data loading
3. **Poor Convergence**: Adjust learning rate or noise probability

### Memory Optimization

```bash
# Use smaller batch size and gradient accumulation
python train_noisy.py \
    --model raft \
    --data.batch_size 2 \
    --trainer.accumulate_grad_batches 4 \
    # ... other parameters
```

## Performance Expectations

Models trained with noise augmentation typically show:

- **Improved robustness** in low-light conditions
- **Better generalization** to real-world noisy scenarios
- **Slight reduction** in performance on clean data (trade-off)
- **Enhanced temporal consistency** in video sequences

## Citation

If you use this noisy training system, please cite the original noise model papers and PTLFlow:

```bibtex
@article{ptlflow,
  title={PTLFlow: A PyTorch Lightning Framework for Optical Flow},
  author={Morimitsu, Henrique},
  journal={arXiv preprint arXiv:2109.05040},
  year={2021}
}
``` 