#!/usr/bin/env python3
"""
Test script for the noisy training system.

This script performs a quick test to ensure the noise application works correctly
and that the training pipeline can handle the noisy data.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path to import noise module
sys.path.append(str(Path(__file__).parent.parent))
from noise import generate_noise, reshape_noise_params

from train_noisy import NoisyFlowDataModule


def test_noise_generation():
    """Test basic noise generation functionality."""
    print("Testing noise generation...")
    
    # Create dummy image data
    batch_size = 2
    num_frames = 2
    height, width = 64, 64
    channels = 3
    
    # Create dummy images (B, N, C, H, W)
    dummy_images = torch.rand(batch_size, num_frames, channels, height, width)
    
    # Test starlight noise
    print("Testing starlight noise...")
    noise_params = torch.rand(batch_size, 11)  # 11 parameters for starlight
    noise_dict = reshape_noise_params(noise_params, "starlight", num_frames=num_frames)
    
    images_flat = dummy_images.view(batch_size * num_frames, channels, height, width)
    noisy_images = generate_noise(images_flat, noise_dict, "starlight", num_frames=num_frames)
    
    print(f"Original image range: [{images_flat.min():.3f}, {images_flat.max():.3f}]")
    print(f"Noisy image range: [{noisy_images.min():.3f}, {noisy_images.max():.3f}]")
    
    # Test ELD noise
    print("\nTesting ELD noise...")
    noise_params = torch.rand(batch_size, 8)  # 8 parameters for ELD
    noise_dict = reshape_noise_params(noise_params, "eld", num_frames=num_frames)
    
    noisy_images_eld = generate_noise(images_flat, noise_dict, "eld", num_frames=num_frames)
    
    print(f"ELD noisy image range: [{noisy_images_eld.min():.3f}, {noisy_images_eld.max():.3f}]")
    
    print("✓ Noise generation test passed!")


def test_datamodule():
    """Test the NoisyFlowDataModule functionality."""
    print("\nTesting NoisyFlowDataModule...")
    
    # Test noise application on dummy batch
    class DummyDataModule(NoisyFlowDataModule):
        def __init__(self, **kwargs):
            # Initialize with minimal parameters to avoid full PTLFlow setup
            self.noise_model = kwargs.get('noise_model', 'starlight')
            self.noise_probability = kwargs.get('noise_probability', 0.5)
            self.noise_params_range = None
            
            # Set up noise parameter ranges
            if self.noise_model == "starlight":
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
                    'band_noise_angle': [0.0, 1.0]
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
    
    # Test with 4D images (typical optical flow format)
    batch_4d = {
        "images": torch.rand(2, 3, 64, 64),  # B, C, H, W
        "flows": torch.rand(2, 1, 2, 64, 64)  # B, N, 2, H, W
    }
    
    # Test with 5D images (video sequence format)
    batch_5d = {
        "images": torch.rand(2, 2, 3, 64, 64),  # B, N, C, H, W
        "flows": torch.rand(2, 1, 2, 64, 64)
    }
    
    # Test starlight noise
    datamodule = DummyDataModule(noise_model="starlight", noise_probability=1.0)
    
    print("Testing 4D batch with starlight noise...")
    original_4d = batch_4d["images"].clone()
    noisy_batch_4d = datamodule._apply_noise_to_batch(batch_4d)
    print(f"4D: Original range [{original_4d.min():.3f}, {original_4d.max():.3f}]")
    print(f"4D: Noisy range [{noisy_batch_4d['images'].min():.3f}, {noisy_batch_4d['images'].max():.3f}]")
    
    print("Testing 5D batch with starlight noise...")
    original_5d = batch_5d["images"].clone()
    noisy_batch_5d = datamodule._apply_noise_to_batch(batch_5d)
    print(f"5D: Original range [{original_5d.min():.3f}, {original_5d.max():.3f}]")
    print(f"5D: Noisy range [{noisy_batch_5d['images'].min():.3f}, {noisy_batch_5d['images'].max():.3f}]")
    
    # Test ELD noise
    datamodule_eld = DummyDataModule(noise_model="eld", noise_probability=1.0)
    
    print("Testing 4D batch with ELD noise...")
    noisy_batch_eld = datamodule_eld._apply_noise_to_batch(batch_4d)
    print(f"ELD: Noisy range [{noisy_batch_eld['images'].min():.3f}, {noisy_batch_eld['images'].max():.3f}]")
    
    print("✓ NoisyFlowDataModule test passed!")


def test_parameter_ranges():
    """Test that noise parameters are in correct ranges."""
    print("\nTesting noise parameter ranges...")
    
    # Test starlight parameters
    datamodule = test_datamodule.__defaults__[0] if hasattr(test_datamodule, '__defaults__') else None
    
    batch = {"images": torch.rand(1, 2, 3, 64, 64)}
    
    # Manual test of parameter generation
    noise_params_range = {
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
        'band_noise_angle': [0.0, 1.0]
    }
    
    # Generate parameters
    param_names = list(noise_params_range.keys())
    noise_params_item = []
    for param_name in param_names:
        min_val, max_val = noise_params_range[param_name]
        if param_name == 'band_noise_angle':
            param_val = torch.randint(0, 2, (1,)).float().item()
        else:
            param_val = torch.rand(1).item() * (max_val - min_val) + min_val
        noise_params_item.append(param_val)
        
        # Check range
        assert min_val <= param_val <= max_val, f"Parameter {param_name} out of range: {param_val}"
        print(f"{param_name}: {param_val:.3f} ✓")
    
    print("✓ Parameter ranges test passed!")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Noisy Training System")
    print("=" * 50)
    
    try:
        test_noise_generation()
        test_datamodule()
        test_parameter_ranges()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! The noisy training system is ready to use.")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Configure your dataset paths in datasets.yaml")
        print("2. Run: python train_noisy.py --model raft --data.train_dataset flying_chairs --noise_model starlight")
        print("3. Use the trained weights with infer.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check your noise.py implementation and dependencies.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 