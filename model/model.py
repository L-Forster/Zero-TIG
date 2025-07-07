import os.path

import torch
import torch.nn as nn
from loss import LossFunction, TextureDifference
from utils.utils import blur, pair_downsampler, viz, warp_tensor, InputPadder
from model.RAFT.raft import RAFT
import ptlflow
from torchvision.transforms.functional import equalize
import torch.nn.functional as F
import numpy as np
import cv2


class SelfEnsemble(nn.Module):
    def __init__(self, model):
        super(SelfEnsemble, self).__init__()
        self.model = model

    def forward(self, x):
        # h-flip
        x_hflip = torch.flip(x, dims=[3])
        out_hflip = self.model(x_hflip)
        out_hflip = torch.flip(out_hflip, dims=[3])

        # v-flip
        x_vflip = torch.flip(x, dims=[2])
        out_vflip = self.model(x_vflip)
        out_vflip = torch.flip(out_vflip, dims=[2])

        # rot90
        x_rot90 = torch.rot90(x, 1, [2, 3])
        out_rot90 = self.model(x_rot90)
        out_rot90 = torch.rot90(out_rot90, -1, [2, 3])

        # original
        out = self.model(x)

        return (out + out_hflip + out_vflip + out_rot90) / 4.0


class Denoise_1(nn.Module):
    def __init__(self, chan_embed=48):
        super(Denoise_1, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(3, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 3, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Denoise_2(nn.Module):
    def __init__(self, chan_embed=96):
        super(Denoise_2, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(12, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 6, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Enhancer(nn.Module):
    def __init__(self, layers, channels):
        super(Enhancer, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        fea = torch.clamp(fea, 0.0001, 1)

        return fea


class Network(nn.Module):

    def __init__(self, args):
        super(Network, self).__init__()

        self.enhance = Enhancer(layers=3, channels=64)
        
        self.use_ensemble = getattr(args, 'use_self_ensemble', True)
        if self.use_ensemble:
            self.denoise_1 = SelfEnsemble(Denoise_1(chan_embed=48))
            self.denoise_2 = SelfEnsemble(Denoise_2(chan_embed=48))
        else:
            self.denoise_1 = Denoise_1(chan_embed=48)
            self.denoise_2 = Denoise_2(chan_embed=48)
            
        self._l2_loss = nn.MSELoss()
        self._l1_loss = nn.L1Loss()
        self.is_WB = True if 'underwater' == args.dataset else False
        self._criterion = LossFunction(self.is_WB)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.TextureDifference = TextureDifference()

        self.last_H3 = None
        self.last_H3_wp = None
        self.last_s3 = None
        self.last_s3_wp = None
        self.is_new_seq = True

        # optical flow
        self.of_model = self.load_optical_flow_model(args, getattr(args, 'of_model_name', 'raft'), getattr(args, 'of_model_path', None))
        self.of_scale = args.of_scale
        
        # bidirectional optical flow settings
        self.use_bidirectional_flow = getattr(args, 'use_bidirectional_flow', True)
        self.occlusion_threshold = getattr(args, 'occlusion_threshold', 1.0)
        self.flow_consistency_alpha = getattr(args, 'flow_consistency_alpha', 0.01)

    def load_optical_flow_model(self, args, model_name='raft', model_path=None):
        """Loads an optical flow model."""
        if model_path:
            # A specific checkpoint is provided. Get the model architecture first.
            model = ptlflow.get_model(model_name)
            
            # Manually load the checkpoint. This is more robust to different formats.
            ckpt = torch.load(model_path, map_location='cpu')
            
            # The checkpoint could be the state_dict itself, or it could be nested.
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            
            # The state_dict may have keys with a 'module.' prefix; remove it.
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:]  # remove 'module.'
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False)
        elif model_name.lower() == 'raft':
            # Default behavior: load the original RAFT model
            model = RAFT(args)
        else:
            # A different model name is given, but no path. Load pretrained from ptlflow.
            model = ptlflow.get_model(model_name)
            
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def cvt_ts2np(self, t):
        # convert tensor to array
        t = t.detach()
        n = t.squeeze().permute((1, 2, 0)).cpu().numpy()
        return n

    def enhance_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def denoise_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
        # if isinstance(m, nn.Conv2d):
        # nn.init.xavier_uniform(m.weight)
        # nn.init.constant(m.bias, 0)

    def forward(self, input):
        eps = 1e-4
        input = input + eps

        L11, L12 = pair_downsampler(input)
        L_pred1 = self.denoise_1(L11)
        L_pred2 = self.denoise_1(L12)
        L2_noise = self.denoise_1(input)
        L2 = input - L2_noise
        L2 = torch.clamp(L2, eps, 1)

        """ concat output from last frm"""
        if self.is_new_seq:
            self.last_H3_wp = torch.zeros_like(L2)
            self.last_s3_wp = torch.zeros_like(L2)
            self.last_H31_wp = torch.zeros_like(L11)
            self.last_H32_wp = torch.zeros_like(L11)
            self.last_s31_wp = torch.zeros_like(L11)
            self.last_s32_wp = torch.zeros_like(L11)
        else:
            # OF + warp
            self.last_H3_wp, self.last_s3_wp = self.update_cache(self.last_H3, self.last_s3, L2.detach())
            self.last_H31_wp, self.last_H32_wp = pair_downsampler(self.last_H3_wp)
            self.last_s31_wp, self.last_s32_wp = pair_downsampler(self.last_s3_wp)

        s2 = self.enhance(torch.cat([self.last_H3_wp, self.last_s3_wp, L2], 1).detach())
        s21, s22 = pair_downsampler(s2)
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)

        H11 = L11 / s21
        H11 = torch.clamp(H11, eps, 1)

        H12 = L12 / s22
        H12 = torch.clamp(H12, eps, 1)

        denoise2_input_1 = torch.cat([self.last_H31_wp, self.last_s31_wp, H11, s21], 1)
        denoise2_subtract_input_1 = torch.cat([H11, s21], 1).detach()
        H3_pred_noise = self.denoise_2(denoise2_input_1)
        H3_pred = denoise2_subtract_input_1 - H3_pred_noise
        H3_pred = torch.clamp(H3_pred, eps, 1)
        H13 = H3_pred[:, :3, :, :]
        s13 = H3_pred[:, 3:, :, :]

        denoise2_input_2 = torch.cat([self.last_H32_wp, self.last_s32_wp, H12, s22], 1)
        denoise2_subtract_input_2 = torch.cat([H12, s22], 1).detach()
        H4_pred_noise = self.denoise_2(denoise2_input_2)
        H4_pred = denoise2_subtract_input_2 - H4_pred_noise
        H4_pred = torch.clamp(H4_pred, eps, 1)
        H14 = H4_pred[:, :3, :, :]
        s14 = H4_pred[:, 3:, :, :]

        denoise2_input_3 = torch.cat([self.last_H3_wp, self.last_s3_wp, H2, s2], 1)
        denoise2_subtract_input_3 = torch.cat([H2, s2], 1).detach()
        H5_pred_noise = self.denoise_2(denoise2_input_3)
        H5_pred = denoise2_subtract_input_3 - H5_pred_noise
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]
        s3 = H5_pred[:, 3:, :, :]

        L_pred1_L_pred2_diff = self.TextureDifference(L_pred1, L_pred2)
        H3_denoised1, H3_denoised2 = pair_downsampler(H3)
        H3_denoised1_H3_denoised2_diff= self.TextureDifference(H3_denoised1, H3_denoised2)

        H1 = L2 / s2
        H1 = torch.clamp(H1, 0, 1)
        H2_blur = blur(H1)
        H3_blur = blur(H3)

        return L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur, H3_denoised1, H3_denoised2

    def _loss(self, input):
        L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur, H3_denoised1, H3_denoised2 = self(
            input)
        loss = 0

        loss += self._criterion(input, L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3,
                                H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur,
                                H3_blur)

        self.update_H3(H3, s3)
        return loss

    def update_H3(self, H3, s3):
        self.last_H3 = H3.detach()
        self.last_s3 = s3.detach()

    def compute_bidirectional_flow(self, img1, img2):
        """
        Compute bidirectional optical flow between two images.
        
        Args:
            img1: Previous frame tensor [B, C, H, W]
            img2: Current frame tensor [B, C, H, W]
            
        Returns:
            flow_forward: Flow from img1 to img2 [B, 2, H, W]
            flow_backward: Flow from img2 to img1 [B, 2, H, W]
            occlusion_mask: Occlusion mask [B, 1, H, W] (1 = occluded, 0 = visible)
        """
        with torch.no_grad():
            # Check if this is the local RAFT model or a ptlflow model
            if hasattr(self.of_model, 'pad'):  # Local RAFT model
                # Forward flow: img1 -> img2
                _, flow_forward = self.of_model(img1, img2, iters=12, test_mode=True)
                # Backward flow: img2 -> img1  
                _, flow_backward = self.of_model(img2, img1, iters=12, test_mode=True)
            else:
                # ptlflow models
                # Forward flow: img1 -> img2
                images_forward = torch.stack([img1, img2], dim=1)  # [B, 2, C, H, W]
                inputs_forward = {'images': images_forward}
                outputs_forward = self.of_model(inputs_forward)
                flow_forward = outputs_forward['flows']
                if flow_forward.dim() == 5:
                    flow_forward = flow_forward.squeeze(1)
                
                # Backward flow: img2 -> img1
                images_backward = torch.stack([img2, img1], dim=1)  # [B, 2, C, H, W]
                inputs_backward = {'images': images_backward}
                outputs_backward = self.of_model(inputs_backward)
                flow_backward = outputs_backward['flows']
                if flow_backward.dim() == 5:
                    flow_backward = flow_backward.squeeze(1)
        
        # Compute occlusion mask using forward-backward consistency
        occlusion_mask = self.compute_occlusion_mask(flow_forward, flow_backward)
        
        return flow_forward, flow_backward, occlusion_mask

    def compute_occlusion_mask(self, flow_forward, flow_backward):
        """
        Compute occlusion mask using forward-backward consistency check.
        
        Args:
            flow_forward: Forward optical flow [B, 2, H, W]
            flow_backward: Backward optical flow [B, 2, H, W]
            
        Returns:
            occlusion_mask: Binary mask [B, 1, H, W] where 1 indicates occlusion
        """
        # Warp backward flow using forward flow
        warped_flow_backward = self.warp_flow(flow_backward, flow_forward)
        
        # Compute forward-backward consistency error
        flow_diff = flow_forward + warped_flow_backward
        consistency_error = torch.norm(flow_diff, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Create occlusion mask based on consistency threshold
        flow_magnitude = torch.norm(flow_forward, dim=1, keepdim=True)
        adaptive_threshold = self.occlusion_threshold + self.flow_consistency_alpha * flow_magnitude
        occlusion_mask = (consistency_error > adaptive_threshold).float()
        
        return occlusion_mask

    def warp_flow(self, flow, warp_flow):
        """
        Warp optical flow using another flow field.
        
        Args:
            flow: Flow to be warped [B, 2, H, W]
            warp_flow: Flow used for warping [B, 2, H, W]
            
        Returns:
            warped_flow: Warped flow [B, 2, H, W]
        """
        B, _, H, W = flow.shape
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=flow.device),
            torch.arange(W, dtype=torch.float32, device=flow.device)
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # Apply warping flow to coordinates
        warped_coords = grid + warp_flow
        
        # Normalize coordinates to [-1, 1] for grid_sample
        warped_coords[:, 0] = 2.0 * warped_coords[:, 0] / (W - 1) - 1.0  # x coordinates
        warped_coords[:, 1] = 2.0 * warped_coords[:, 1] / (H - 1) - 1.0  # y coordinates
        
        # Reshape for grid_sample: [B, H, W, 2]
        warped_coords = warped_coords.permute(0, 2, 3, 1)
        
        # Warp the flow
        warped_flow = F.grid_sample(flow, warped_coords, mode='bilinear', 
                                   padding_mode='zeros', align_corners=True)
        
        return warped_flow

    def bidirectional_warp_tensor(self, flow_forward, flow_backward, occlusion_mask, 
                                 tensor_prev, tensor_curr):
        """
        Perform bidirectional warping with occlusion handling.
        
        Args:
            flow_forward: Forward flow [B, 2, H_flow, W_flow]
            flow_backward: Backward flow [B, 2, H_flow, W_flow]
            occlusion_mask: Occlusion mask [B, 1, H_flow, W_flow]
            tensor_prev: Previous frame tensor to warp [B, C, H, W]
            tensor_curr: Current frame tensor [B, C, H, W]
            
        Returns:
            warped_tensor: Warped tensor with occlusion handling [B, C, H, W]
            final_occlusion_mask: Occlusion mask at tensor resolution [B, 1, H, W]
        """
        B, C, H, W = tensor_prev.shape
        _, _, H_flow, W_flow = flow_forward.shape
        
        # Resize flows and occlusion mask to match tensor resolution if needed
        if H != H_flow or W != W_flow:
            flow_forward_resized = F.interpolate(flow_forward, (H, W), mode='bilinear', align_corners=True)
            flow_forward_resized = flow_forward_resized * torch.tensor([W / W_flow, H / H_flow], 
                                                                      device=flow_forward.device).view(1, 2, 1, 1)
            
            flow_backward_resized = F.interpolate(flow_backward, (H, W), mode='bilinear', align_corners=True)
            flow_backward_resized = flow_backward_resized * torch.tensor([W / W_flow, H / H_flow], 
                                                                        device=flow_backward.device).view(1, 2, 1, 1)
            
            occlusion_mask_resized = F.interpolate(occlusion_mask, (H, W), mode='bilinear', align_corners=True)
            occlusion_mask_resized = (occlusion_mask_resized > 0.5).float()
        else:
            flow_forward_resized = flow_forward
            flow_backward_resized = flow_backward
            occlusion_mask_resized = occlusion_mask
        
        # Forward warp: warp previous frame to current frame
        warped_prev, _ = warp_tensor(flow_forward_resized, tensor_prev, tensor_curr)
        
        # Backward warp: warp current frame to previous frame, then forward again
        # This can help fill occlusions with information from current frame
        warped_curr_to_prev, _ = warp_tensor(flow_backward_resized, tensor_curr, tensor_prev)
        warped_curr_back, _ = warp_tensor(flow_forward_resized, warped_curr_to_prev, tensor_curr)
        
        # Combine warped tensors using occlusion mask
        # Use forward warp in non-occluded regions, fallback to backward warp in occluded regions
        visibility_mask = 1.0 - occlusion_mask_resized  # 1 = visible, 0 = occluded
        
        # Weight the contributions
        alpha = 0.8  # Weight for forward warp
        beta = 0.2   # Weight for backward warp
        
        warped_tensor = (alpha * visibility_mask * warped_prev + 
                        beta * visibility_mask * warped_curr_back + 
                        occlusion_mask_resized * tensor_curr)  # Use current frame for occluded regions
        
        return warped_tensor, occlusion_mask_resized

    def update_cache(self, last_H3, last_s3, L2):
        if not self.use_bidirectional_flow:
            # Use original unidirectional method
            return self.update_cache_unidirectional(last_H3, last_s3, L2)
        
        # 0. resize
        ht_org, wd_org = last_H3[0].shape[-2:]
        ht = ht_org // self.of_scale
        wd = wd_org // self.of_scale
        last_H3_tmp = F.interpolate(last_H3, (ht, wd), mode='bilinear')
        L2_tmp = F.interpolate(L2, (ht, wd), mode='bilinear')

        # 1. Equalize the histogram
        last_H3_tmp = last_H3_tmp * 255
        last_H3_tmp = last_H3_tmp.to(torch.float32)

        L2_tmp = equalize((L2_tmp * 255).to(torch.uint8))
        L2_tmp = L2_tmp.to(torch.float32)

        # 2. Ensure RAFT model is on the same device
        self.of_model = self.of_model.to(L2.device)
        self.of_model.eval()

        # 3. Compute bidirectional optical flow
        flow_forward, flow_backward, occlusion_mask = self.compute_bidirectional_flow(last_H3_tmp, L2_tmp)

        # 4. Bidirectional warping with occlusion handling
        warped_tensor_H3, final_occlusion_mask = self.bidirectional_warp_tensor(
            flow_forward, flow_backward, occlusion_mask, last_H3, L2)
        
        warped_tensor_s3, _ = self.bidirectional_warp_tensor(
            flow_forward, flow_backward, occlusion_mask, last_s3, L2)

        return warped_tensor_H3, warped_tensor_s3

    def update_cache_unidirectional(self, last_H3, last_s3, L2):
        """Original unidirectional optical flow method for backward compatibility."""
        # 0. resize
        ht_org, wd_org = last_H3[0].shape[-2:]
        ht = ht_org // self.of_scale
        wd = wd_org // self.of_scale
        last_H3_tmp = F.interpolate(last_H3, (ht, wd), mode='bilinear')
        L2_tmp = F.interpolate(L2, (ht, wd), mode='bilinear')

        # 1. Equalize the histogram
        last_H3_tmp = last_H3_tmp * 255
        last_H3_tmp = last_H3_tmp.to(torch.float32)

        L2_tmp = equalize((L2_tmp * 255).to(torch.uint8))
        L2_tmp = L2_tmp.to(torch.float32)

        # 2. OF last->this
        # Ensure RAFT model is on the same device as the inputs
        self.of_model = self.of_model.to(L2.device)
        self.of_model.eval()  # Ensure model is in eval mode
        
        # Use RAFT forward method
        with torch.no_grad():
            # Check if this is the local RAFT model or a ptlflow model
            if hasattr(self.of_model, 'pad'):  # Local RAFT model has a 'pad' method
                # Local RAFT expects two separate image tensors
                _, flow_up = self.of_model(last_H3_tmp, L2_tmp, iters=12, test_mode=True)
            else:
                # ptlflow models expect a dictionary with stacked images
                images_stacked = torch.stack([last_H3_tmp, L2_tmp], dim=1)  # [B, 2, C, H, W]
                inputs_dict = {'images': images_stacked}
                outputs = self.of_model(inputs_dict)
                flow_up = outputs['flows']
                # ptlflow models may return a 5D tensor [B, T, C, H, W]. T=1 for 2 images.
                if flow_up.dim() == 5:
                    flow_up = flow_up.squeeze(1)

        # 3. Warp
        warped_tensor_H3, overlap_tensor = warp_tensor(flow_up, last_H3, L2)
        warped_tensor_s3, _ = warp_tensor(flow_up, last_s3, L2)

        return warped_tensor_H3, warped_tensor_s3


class Finetunemodel(nn.Module):

    def __init__(self, args):
        super(Finetunemodel, self).__init__()
        self.args = args

        self.use_ensemble = getattr(args, 'use_self_ensemble', True)
        if self.use_ensemble:
            self.denoise_1 = SelfEnsemble(Denoise_1(chan_embed=48))
            self.denoise_2 = SelfEnsemble(Denoise_2(chan_embed=48))
        else:
            self.denoise_1 = Denoise_1(chan_embed=48)
            self.denoise_2 = Denoise_2(chan_embed=48)

        self.enhance = Enhancer(layers=3, channels=64)
        base_weights = torch.load(args.model_pretrain, map_location='cuda:0')
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        self.last_H3 = None
        self.last_H3_wp = None
        self.last_s3 = None
        self.last_s3_wp = None
        self.is_new_seq = True

        # optical flow
        self.of_model = self.load_optical_flow_model(args, getattr(args, 'of_model_name', 'raft'), getattr(args, 'of_model_path', None))
        self.of_scale = args.of_scale

        # bidirectional optical flow settings
        self.use_bidirectional_flow = getattr(args, 'use_bidirectional_flow', True)
        self.occlusion_threshold = getattr(args, 'occlusion_threshold', 1.0)
        self.flow_consistency_alpha = getattr(args, 'flow_consistency_alpha', 0.01)

    def load_optical_flow_model(self, args, model_name='raft', model_path=None):
        """Loads an optical flow model."""
        if model_path:
            # A specific checkpoint is provided. Get the model architecture first.
            model = ptlflow.get_model(model_name)
            
            # Manually load the checkpoint. This is more robust to different formats.
            ckpt = torch.load(model_path, map_location='cpu')
            
            # The checkpoint could be the state_dict itself, or it could be nested.
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            
            # The state_dict may have keys with a 'module.' prefix; remove it.
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:]  # remove 'module.'
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False)
        elif model_name.lower() == 'raft':
            # Default behavior: load the original RAFT model
            model = RAFT(args)
        else:
            # A different model name is given, but no path. Load pretrained from ptlflow.
            model = ptlflow.get_model(model_name)
            
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def cvt_ts2np(self, t):
        # convert tensor to array
        t = t.detach()
        n = t.squeeze().permute((1, 2, 0)).cpu().numpy()
        return n

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def compute_bidirectional_flow(self, img1, img2):
        """
        Compute bidirectional optical flow between two images.
        
        Args:
            img1: Previous frame tensor [B, C, H, W]
            img2: Current frame tensor [B, C, H, W]
            
        Returns:
            flow_forward: Flow from img1 to img2 [B, 2, H, W]
            flow_backward: Flow from img2 to img1 [B, 2, H, W]
            occlusion_mask: Occlusion mask [B, 1, H, W] (1 = occluded, 0 = visible)
        """
        with torch.no_grad():
            # Check if this is the local RAFT model or a ptlflow model
            if hasattr(self.of_model, 'pad'):  # Local RAFT model
                # Forward flow: img1 -> img2
                _, flow_forward = self.of_model(img1, img2, iters=12, test_mode=True)
                # Backward flow: img2 -> img1  
                _, flow_backward = self.of_model(img2, img1, iters=12, test_mode=True)
            else:
                # ptlflow models
                # Forward flow: img1 -> img2
                images_forward = torch.stack([img1, img2], dim=1)  # [B, 2, C, H, W]
                inputs_forward = {'images': images_forward}
                outputs_forward = self.of_model(inputs_forward)
                flow_forward = outputs_forward['flows']
                if flow_forward.dim() == 5:
                    flow_forward = flow_forward.squeeze(1)
                
                # Backward flow: img2 -> img1
                images_backward = torch.stack([img2, img1], dim=1)  # [B, 2, C, H, W]
                inputs_backward = {'images': images_backward}
                outputs_backward = self.of_model(inputs_backward)
                flow_backward = outputs_backward['flows']
                if flow_backward.dim() == 5:
                    flow_backward = flow_backward.squeeze(1)
        
        # Compute occlusion mask using forward-backward consistency
        occlusion_mask = self.compute_occlusion_mask(flow_forward, flow_backward)
        
        return flow_forward, flow_backward, occlusion_mask

    def compute_occlusion_mask(self, flow_forward, flow_backward):
        """
        Compute occlusion mask using forward-backward consistency check.
        
        Args:
            flow_forward: Forward optical flow [B, 2, H, W]
            flow_backward: Backward optical flow [B, 2, H, W]
            
        Returns:
            occlusion_mask: Binary mask [B, 1, H, W] where 1 indicates occlusion
        """
        # Warp backward flow using forward flow
        warped_flow_backward = self.warp_flow(flow_backward, flow_forward)
        
        # Compute forward-backward consistency error
        flow_diff = flow_forward + warped_flow_backward
        consistency_error = torch.norm(flow_diff, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Create occlusion mask based on consistency threshold
        flow_magnitude = torch.norm(flow_forward, dim=1, keepdim=True)
        adaptive_threshold = self.occlusion_threshold + self.flow_consistency_alpha * flow_magnitude
        occlusion_mask = (consistency_error > adaptive_threshold).float()
        
        return occlusion_mask

    def warp_flow(self, flow, warp_flow):
        """
        Warp optical flow using another flow field.
        
        Args:
            flow: Flow to be warped [B, 2, H, W]
            warp_flow: Flow used for warping [B, 2, H, W]
            
        Returns:
            warped_flow: Warped flow [B, 2, H, W]
        """
        B, _, H, W = flow.shape
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=flow.device),
            torch.arange(W, dtype=torch.float32, device=flow.device)
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # Apply warping flow to coordinates
        warped_coords = grid + warp_flow
        
        # Normalize coordinates to [-1, 1] for grid_sample
        warped_coords[:, 0] = 2.0 * warped_coords[:, 0] / (W - 1) - 1.0  # x coordinates
        warped_coords[:, 1] = 2.0 * warped_coords[:, 1] / (H - 1) - 1.0  # y coordinates
        
        # Reshape for grid_sample: [B, H, W, 2]
        warped_coords = warped_coords.permute(0, 2, 3, 1)
        
        # Warp the flow
        warped_flow = F.grid_sample(flow, warped_coords, mode='bilinear', 
                                   padding_mode='zeros', align_corners=True)
        
        return warped_flow

    def bidirectional_warp_tensor(self, flow_forward, flow_backward, occlusion_mask, 
                                 tensor_prev, tensor_curr):
        """
        Perform bidirectional warping with occlusion handling.
        
        Args:
            flow_forward: Forward flow [B, 2, H_flow, W_flow]
            flow_backward: Backward flow [B, 2, H_flow, W_flow]
            occlusion_mask: Occlusion mask [B, 1, H_flow, W_flow]
            tensor_prev: Previous frame tensor to warp [B, C, H, W]
            tensor_curr: Current frame tensor [B, C, H, W]
            
        Returns:
            warped_tensor: Warped tensor with occlusion handling [B, C, H, W]
            final_occlusion_mask: Occlusion mask at tensor resolution [B, 1, H, W]
        """
        B, C, H, W = tensor_prev.shape
        _, _, H_flow, W_flow = flow_forward.shape
        
        # Resize flows and occlusion mask to match tensor resolution if needed
        if H != H_flow or W != W_flow:
            flow_forward_resized = F.interpolate(flow_forward, (H, W), mode='bilinear', align_corners=True)
            flow_forward_resized = flow_forward_resized * torch.tensor([W / W_flow, H / H_flow], 
                                                                      device=flow_forward.device).view(1, 2, 1, 1)
            
            flow_backward_resized = F.interpolate(flow_backward, (H, W), mode='bilinear', align_corners=True)
            flow_backward_resized = flow_backward_resized * torch.tensor([W / W_flow, H / H_flow], 
                                                                        device=flow_backward.device).view(1, 2, 1, 1)
            
            occlusion_mask_resized = F.interpolate(occlusion_mask, (H, W), mode='bilinear', align_corners=True)
            occlusion_mask_resized = (occlusion_mask_resized > 0.5).float()
        else:
            flow_forward_resized = flow_forward
            flow_backward_resized = flow_backward
            occlusion_mask_resized = occlusion_mask
        
        # Forward warp: warp previous frame to current frame
        warped_prev, _ = warp_tensor(flow_forward_resized, tensor_prev, tensor_curr)
        
        # Backward warp: warp current frame to previous frame, then forward again
        # This can help fill occlusions with information from current frame
        warped_curr_to_prev, _ = warp_tensor(flow_backward_resized, tensor_curr, tensor_prev)
        warped_curr_back, _ = warp_tensor(flow_forward_resized, warped_curr_to_prev, tensor_curr)
        
        # Combine warped tensors using occlusion mask
        # Use forward warp in non-occluded regions, fallback to backward warp in occluded regions
        visibility_mask = 1.0 - occlusion_mask_resized  # 1 = visible, 0 = occluded
        
        # Weight the contributions
        alpha = 0.8  # Weight for forward warp
        beta = 0.2   # Weight for backward warp
        
        warped_tensor = (alpha * visibility_mask * warped_prev + 
                        beta * visibility_mask * warped_curr_back + 
                        occlusion_mask_resized * tensor_curr)  # Use current frame for occluded regions
        
        return warped_tensor, occlusion_mask_resized

    def forward(self, input):
        eps = 1e-4
        input = input + eps
        L2_noise = self.denoise_1(input)
        L2 = input - L2_noise
        L2 = torch.clamp(L2, eps, 1)

        """ concat output from last frm"""
        if self.is_new_seq:
            self.last_H3_wp = torch.zeros_like(L2)
            self.last_s3_wp = torch.zeros_like(L2)
        else:
            # OF + warp
            self.last_H3_wp, self.last_s3_wp = self.update_cache(self.last_H3, self.last_s3, L2.detach())

        s2 = self.enhance(torch.cat([self.last_H3_wp, self.last_s3_wp, L2], 1).detach())
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)

        if self.is_new_seq:
            self.last_H3_wp = H2.detach()
            self.last_s3_wp = H2.detach()

        denoise2_input = torch.cat([self.last_H3_wp, self.last_s3_wp, H2, s2], 1)
        denoise2_subtract_input = torch.cat([H2, s2], 1).detach()
        H5_pred_noise = self.denoise_2(denoise2_input)
        H5_pred = denoise2_subtract_input - H5_pred_noise
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]
        s3 = H5_pred[:, 3:, :, :]

        self.update_H3(H3, s3)
        return H2,H3,s3

    def update_H3(self, H3, s3):
        self.last_H3 = H3.detach()
        self.last_s3 = s3.detach()

    def update_cache(self, last_H3, last_s3, L2):
        if not self.use_bidirectional_flow:
            # Use original unidirectional method
            return self.update_cache_unidirectional(last_H3, last_s3, L2)
        
        # 0. resize
        ht_org, wd_org = last_H3[0].shape[-2:]
        ht = ht_org // self.of_scale
        wd = wd_org // self.of_scale
        last_H3_tmp = F.interpolate(last_H3, (ht, wd), mode='bilinear')
        L2_tmp = F.interpolate(L2, (ht, wd), mode='bilinear')

        # 1. Equalize the histogram
        last_H3_tmp = last_H3_tmp * 255
        last_H3_tmp = last_H3_tmp.to(torch.float32)

        L2_tmp = equalize((L2_tmp * 255).to(torch.uint8))
        L2_tmp = L2_tmp.to(torch.float32)

        # 2. Ensure RAFT model is on the same device
        self.of_model = self.of_model.to(L2.device)
        self.of_model.eval()

        # 3. Compute bidirectional optical flow
        flow_forward, flow_backward, occlusion_mask = self.compute_bidirectional_flow(last_H3_tmp, L2_tmp)

        # 4. Bidirectional warping with occlusion handling
        warped_tensor_H3, final_occlusion_mask = self.bidirectional_warp_tensor(
            flow_forward, flow_backward, occlusion_mask, last_H3, L2)
        
        warped_tensor_s3, _ = self.bidirectional_warp_tensor(
            flow_forward, flow_backward, occlusion_mask, last_s3, L2)

        return warped_tensor_H3, warped_tensor_s3

    def update_cache_unidirectional(self, last_H3, last_s3, L2):
        """Original unidirectional optical flow method for backward compatibility."""
        # 0. resize
        ht_org, wd_org = last_H3[0].shape[-2:]
        ht = ht_org // self.of_scale
        wd = wd_org // self.of_scale
        last_H3_tmp = F.interpolate(last_H3, (ht, wd), mode='bilinear')
        L2_tmp = F.interpolate(L2, (ht, wd), mode='bilinear')

        # 1. Equalize the histogram
        last_H3_tmp = last_H3_tmp * 255
        last_H3_tmp = last_H3_tmp.to(torch.float32)

        L2_tmp = equalize((L2_tmp * 255).to(torch.uint8))
        L2_tmp = L2_tmp.to(torch.float32)

        # 2. OF last->this
        # Ensure RAFT model is on the same device as the inputs
        self.of_model = self.of_model.to(L2.device)
        self.of_model.eval()  # Ensure model is in eval mode
        
        # Use RAFT forward method
        with torch.no_grad():
            # Check if this is the local RAFT model or a ptlflow model
            if hasattr(self.of_model, 'pad'):  # Local RAFT model has a 'pad' method
                # Local RAFT expects two separate image tensors
                _, flow_up = self.of_model(last_H3_tmp, L2_tmp, iters=12, test_mode=True)
            else:
                # ptlflow models expect a dictionary with stacked images
                images_stacked = torch.stack([last_H3_tmp, L2_tmp], dim=1)  # [B, 2, C, H, W]
                inputs_dict = {'images': images_stacked}
                outputs = self.of_model(inputs_dict)
                flow_up = outputs['flows']
                # ptlflow models may return a 5D tensor [B, T, C, H, W]. T=1 for 2 images.
                if flow_up.dim() == 5:
                    flow_up = flow_up.squeeze(1)

        # 3. Warp
        warped_tensor_H3, overlap_tensor = warp_tensor(flow_up, last_H3, L2)
        warped_tensor_s3, _ = warp_tensor(flow_up, last_s3, L2)

        return warped_tensor_H3, warped_tensor_s3