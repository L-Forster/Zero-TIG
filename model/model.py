import os.path

import torch
import torch.nn as nn
from loss import LossFunction, TextureDifference
from utils.utils import blur, pair_downsampler, viz, warp_tensor, InputPadder
from model.RAFT.raft import RAFT
from torchvision.transforms.functional import equalize
import torch.nn.functional as F
import numpy as np
import cv2

from model.RAFT.raft import RAFT # Assuming this is correctly pathed
from .adaptive_sampler import AdaptiveDownsamplerSTN # Import the new module


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
        self.denoise_1 = Denoise_1(chan_embed=48)
        self.denoise_2 = Denoise_2(chan_embed=48)
        self._l2_loss = nn.MSELoss()
        self._l1_loss = nn.L1Loss()
        self.is_WB = True if 'underwater' in args.dataset.lower() else False 
        self._criterion = LossFunction(self.is_WB)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.TextureDifference = TextureDifference()

        self.last_H3 = None
        self.last_H3_wp = None
        self.last_s3 = None
        self.last_s3_wp = None
        self.is_new_seq = True

        self.adaptive_downsampler = AdaptiveDownsamplerSTN(
            output_size_reduction_factor=args.of_scale,
            min_stn_scale=1.0 / (args.of_scale + 1.0), # e.g., if of_scale=3, min_s = 0.25 (for 4x effective zoom out)
            max_stn_scale=1.0 / max(1.5, args.of_scale - 1.0), # e.g., if of_scale=3, max_s = 0.5 (for 2x effective zoom out)
            device=torch.device("cuda" if args.cuda else "cpu")
        )
        self.padder = None # Initialize padder, will be created on first use if needed by RAFT


        # optical flow
        self.raft = self.load_raft(args)

    def load_raft(self, args):
        raft = torch.nn.DataParallel(RAFT(args))
        # load pre-trained data
        raft.load_state_dict(torch.load(args.raft_model))
        raft = raft.module
        raft.eval()
        for param in raft.parameters():
            param.requires_grad = False
        return raft

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
        L_pred1 = L11 - self.denoise_1(L11)
        L_pred2 = L12 - self.denoise_1(L12)
        L2 = input - self.denoise_1(input)
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

        H3_pred = torch.cat([H11, s21], 1).detach() - self.denoise_2(torch.cat([self.last_H31_wp, self.last_s31_wp, H11, s21], 1))
        H3_pred = torch.clamp(H3_pred, eps, 1)
        H13 = H3_pred[:, :3, :, :]
        s13 = H3_pred[:, 3:, :, :]

        H4_pred = torch.cat([H12, s22], 1).detach() - self.denoise_2(torch.cat([self.last_H32_wp, self.last_s32_wp, H12, s22], 1))
        H4_pred = torch.clamp(H4_pred, eps, 1)
        H14 = H4_pred[:, :3, :, :]
        s14 = H4_pred[:, 3:, :, :]

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([self.last_H3_wp, self.last_s3_wp, H2, s2], 1))
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

    def update_cache(self, last_H3, last_s3, L2):
        # `last_H3_cache` and `last_s3_cache` are the stored self.last_H3, self.last_s3
        # `current_L2_detached` is L2.detach() from the current frame

        # Adaptive downsampling
        # last_H3_tmp will be R(t-1)_RD downsampled
        # L2_tmp will be I_t_LD (denoised current input) downsampled
        last_H3_tmp, _ = self.adaptive_downsampler(last_H3_cache)
        L2_tmp, _ = self.adaptive_downsampler(current_L2_detached)

        # Preprocessing for RAFT input based on Zero-TIG paper
        # Input 1 to RAFT: R(t-1)_RD (last_H3_tmp). Paper implies it's used directly or after simple scaling.
        # Your original code scaled it to [0, 255] then float. Let's assume RAFT handles [0,1] for now.
        # If RAFT expects [0,255], scaling is needed: raft_input1 = last_H3_tmp * 255
        raft_input1 = last_H3_tmp

        # Input 2 to RAFT: HE(I_t_LD) (L2_tmp after HE)
        # Ensure L2_tmp is in [0,1] before scaling to uint8 for equalize
        L2_tmp_clamped = torch.clamp(L2_tmp, 0, 1)
        L2_tmp_uint8 = (L2_tmp_clamped * 255).to(torch.uint8)
        raft_input2 = equalize(L2_tmp_uint8).to(torch.float32) / 255.0

        # Pad inputs if RAFT requires specific divisibility (e.g., by 8)
        # This should be done *after* adaptive downsampling and HE.
        if self.padder is None or self.padder.original_shape != raft_input1.shape:
             # Assuming RAFT needs divisibility by 8, adjust as needed
            self.padder = InputPadder(raft_input1.shape, divis_by=8)
        
        raft_input1_padded, raft_input2_padded = self.padder.pad(raft_input1, raft_input2)

        # 2. Optical Flow Computation
        # RAFT usually expects inputs in range [0, 255] or normalized to [-1, 1].
        # The original RAFT implementation often does internal normalization.
        # Let's assume current RAFT wrapper handles [0,1] or does its own normalization.
        # If it strictly needs [0,255]:
        #   raft_input1_final = raft_input1_padded * 255
        #   raft_input2_final = raft_input2_padded * 255
        # Or for [-1,1]:
        #   raft_input1_final = raft_input1_padded * 2.0 - 1.0
        #   raft_input2_final = raft_input2_padded * 2.0 - 1.0
        # For now, assuming [0,1] is fine for the RAFT call.
        _, flow_predicted_padded = self.raft(raft_input1_padded, raft_input2_padded, iters=20, test_mode=True)
        
        flow_predicted = self.padder.unpad(flow_predicted_padded) # Unpad the flow

        # 3. Upsample flow and Warp
        original_ht, original_wd = last_H3_cache.shape[-2:] # Target size for warping is original full-res

        flow_upsampled = F.interpolate(flow_predicted, size=(original_ht, original_wd), mode='bilinear', align_corners=False)
        # Scale flow vectors
        scale_w = float(original_wd) / float(flow_predicted.shape[3]) # wd_orig / wd_small
        scale_h = float(original_ht) / float(flow_predicted.shape[2]) # ht_orig / ht_small
        flow_upsampled[:, 0, :, :] *= scale_w
        flow_upsampled[:, 1, :, :] *= scale_h

        # Warp original full-resolution tensors using the upscaled and scaled flow
        warped_H3, _ = warp_tensor(flow_upsampled, last_H3_cache, current_L2_detached) # current_L2_detached for target grid
        warped_s3, _ = warp_tensor(flow_upsampled, last_s3_cache, current_L2_detached)

        return warped_H3, warped_s3

class Finetunemodel(nn.Module):

    def __init__(self, args):
        super(Finetunemodel, self).__init__()

        self.enhance = Enhancer(layers=3, channels=64)
        self.denoise_1 = Denoise_1(chan_embed=48)
        self.denoise_2 = Denoise_2(chan_embed=48)

        weights = args.model_pretrain
        base_weights = torch.load(weights, map_location='cuda:0')
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

        # Initialize Adaptive Downsampler for Optical Flow
        self.adaptive_downsampler = AdaptiveDownsamplerSTN(
            output_size_reduction_factor=args.of_scale,
            min_stn_scale=1.0 / (args.of_scale + 1.0),
            max_stn_scale=1.0 / max(1.5, args.of_scale - 1.0),
            device=torch.device("cuda" if args.cuda else "cpu") # Add args.cuda check
        )
        self.padder = None # Initialize padder for RAFT inputs


        # optical flow
        self.raft = self.load_raft(args)


    def load_raft(self, args):
        raft = torch.nn.DataParallel(RAFT(args))
        # load pre-trained data
        raft.load_state_dict(torch.load(args.raft_model))
        raft = raft.module
        raft.eval()
        for param in raft.parameters():
            param.requires_grad = False
        return raft


    def cvt_ts2np(self, t):
            t = t.detach()
            n = t.squeeze().permute((1, 2, 0)).cpu().numpy()
            return n

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        eps = 1e-4
        input = input + eps
        L2 = input - self.denoise_1(input)
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

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([self.last_H3_wp, self.last_s3_wp, H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]
        s3 = H5_pred[:, 3:, :, :]

        self.update_H3(H3, s3)
        return H2,H3,s3

    def update_H3(self, H3, s3):
        self.last_H3 = H3.detach()
        self.last_s3 = s3.detach()

    def update_cache(self, last_H3_cache, last_s3_cache, current_L2_detached):
        # Adaptive downsampling
        last_H3_tmp, _ = self.adaptive_downsampler(last_H3_cache)
        L2_tmp, _ = self.adaptive_downsampler(current_L2_detached)

        # Preprocessing for RAFT input
        raft_input1 = last_H3_tmp
        L2_tmp_clamped = torch.clamp(L2_tmp, 0, 1)
        L2_tmp_uint8 = (L2_tmp_clamped * 255).to(torch.uint8)
        raft_input2 = equalize(L2_tmp_uint8).to(torch.float32) / 255.0
        
        # Pad inputs if RAFT requires specific divisibility
        if self.padder is None or self.padder.original_shape != raft_input1.shape:
            self.padder = InputPadder(raft_input1.shape, divis_by=8) # Assuming InputPadder is accessible
        
        raft_input1_padded, raft_input2_padded = self.padder.pad(raft_input1, raft_input2)

        # 2. Optical Flow Computation
        _, flow_predicted_padded = self.raft(raft_input1_padded, raft_input2_padded, iters=20, test_mode=True)
        flow_predicted = self.padder.unpad(flow_predicted_padded)

        # 3. Upsample flow and Warp
        original_ht, original_wd = last_H3_cache.shape[-2:]
        flow_upsampled = F.interpolate(flow_predicted, size=(original_ht, original_wd), mode='bilinear', align_corners=False)
        scale_w = float(original_wd) / float(flow_predicted.shape[3])
        scale_h = float(original_ht) / float(flow_predicted.shape[2])
        flow_upsampled[:, 0, :, :] *= scale_w
        flow_upsampled[:, 1, :, :] *= scale_h

        warped_H3, _ = warp_tensor(flow_upsampled, last_H3_cache, current_L2_detached)
        warped_s3, _ = warp_tensor(flow_upsampled, last_s3_cache, current_L2_detached)

        return warped_H3, warped_s3
        