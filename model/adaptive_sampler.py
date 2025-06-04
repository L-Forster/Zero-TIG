# --- Create a new file, e.g., model/adaptive_sampler.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms.functional import normalize as tv_normalize # For ImageNet normalization

class AdaptiveDownsamplerSTN(nn.Module):
    def __init__(self, output_size_reduction_factor=3.0,
                 min_stn_scale=0.25, # Corresponds to 4x zoom-out if output grid is 1x
                 max_stn_scale=0.75, # Corresponds to ~1.33x zoom-out
                 device='cuda'):
        super(AdaptiveDownsamplerSTN, self).__init__()
        self.output_size_reduction_factor = float(output_size_reduction_factor)
        self.min_stn_scale = float(min_stn_scale) # Min scale for STN's theta (smaller = more zoom out)
        self.max_stn_scale = float(max_stn_scale) # Max scale for STN's theta (larger = less zoom out)
        self.device = device

        # Load a lightweight pre-trained model (ResNet18)
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Use only the initial conv layers and layer1
        self.feature_extractor = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1 # Output of layer1 is 64 channels
        ).to(self.device)

        # Freeze the pre-trained feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Small learnable head to predict a single scale control parameter (alpha)
        self.scale_alpha_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 32, kernel_size=1), # Input channels from ResNet18.layer1
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() # Output alpha between 0 and 1
        ).to(self.device)

        # ImageNet normalization parameters
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _normalize_imagenet(self, x):
        # x is assumed to be Bx3xHxW in [0,1] range
        return (x - self.imagenet_mean) / self.imagenet_std

    def forward(self, x_orig):
        # x_orig is the input image (e.g., last_H3 or L2 from your Network)
        b, c_in, ht_org, wd_org = x_orig.shape

        # Prepare input for ResNet18 (3 channels, normalized)
        if c_in == 1:
            x_feat_input = x_orig.repeat(1, 3, 1, 1) # Grayscale to RGB
        elif c_in == 3:
            x_feat_input = x_orig
        else:
            # Fallback or error for unexpected channel count
            # For simplicity, trying to use first 3 channels if more, or error
            if c_in > 3:
                x_feat_input = x_orig[:, :3, :, :]
            else: # c_in == 2 or other
                 raise ValueError(f"AdaptiveDownsamplerSTN expects 1 or 3 input channels, got {c_in}")

        x_normalized = self._normalize_imagenet(x_feat_input)

        self.feature_extractor.eval() # Ensure pre-trained part is in eval mode
        with torch.no_grad(): # No gradients for the frozen feature extractor
            features = self.feature_extractor(x_normalized)

        # Predict alpha (0 to 1) for STN scaling
        alpha = self.scale_alpha_predictor(features).squeeze(-1).squeeze(-1) # Shape: (B, 1) or (B)

        # Interpolate STN scaling factor (s)
        # alpha = 0 -> s = min_stn_scale (e.g., 0.25, max zoom-out)
        # alpha = 1 -> s = max_stn_scale (e.g., 0.5, less zoom-out)
        stn_s = self.min_stn_scale + alpha * (self.max_stn_scale - self.min_stn_scale) # Shape: (B,1)

        # Create affine transformation matrix (theta) for pure scaling
        # theta = [[s, 0, 0], [0, s, 0]]
        theta = torch.zeros(b, 2, 3, device=self.device, dtype=x_orig.dtype)
        theta[:, 0, 0] = stn_s.squeeze()
        theta[:, 1, 1] = stn_s.squeeze()

        # Define the output grid size for F.grid_sample
        # This determines the actual size reduction for RAFT
        ht_out = ht_org // int(self.output_size_reduction_factor)
        wd_out = wd_org // int(self.output_size_reduction_factor)
        output_grid_size = torch.Size([b, c_in, ht_out, wd_out]) # Use c_in from original x_orig

        # Perform affine transformation (scaling)
        grid = F.affine_grid(theta, output_grid_size, align_corners=False)
        x_downsampled = F.grid_sample(x_orig, grid, mode='bilinear', align_corners=False)

        # The effective "zoom" applied by STN before sampling to the fixed output grid
        # is 1/stn_s. A larger 1/stn_s means more of the original image content
        # is squeezed into the output grid (more apparent downsampling).
        # A smaller 1/stn_s means less of the original is squeezed (less apparent downsampling).
        # For user, "effective_downsampling_factor" is more intuitive.
        effective_downsampling_factor = 1.0 / stn_s

        return x_downsampled, effective_downsampling_factor