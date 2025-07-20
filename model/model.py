import os.path
import torch
import torch.nn as nn
from loss import LossFunction, TextureDifference
from utils.utils import blur, pair_downsampler, viz, warp_tensor, InputPadder, get_next_frame_path
from model.RAFT.raft import RAFT
import ptlflow
from torchvision.transforms.functional import equalize, to_tensor
from PIL import Image
import torch.nn.functional as F
import numpy as np
import cv2
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
# --- UNCHANGED MODULES ---

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
        kernel_size, dilation = 3, 1
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
        self.blocks = nn.ModuleList([self.conv for _ in range(layers)])
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        return torch.clamp(fea, 0.0001, 1)

# --- NETWORK WITH BIDIRECTIONAL WARPING IMPLEMENTED ---

class Network(nn.Module):

    def __init__(self, args):
        super(Network, self).__init__()

        self.enhance = Enhancer(layers=3, channels=64)
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

        # Optical Flow settings
        self.of_model = self.load_optical_flow_model(args, getattr(args, 'of_model_name', 'raft'), getattr(args, 'of_model_path', None))
        
        # Load a second OF model for the backward pass if specified
        if hasattr(args, 'of_model_path_bwd') and args.of_model_path_bwd:
            self.of_model_bwd = self.load_optical_flow_model(args, getattr(args, 'of_model_name_bwd', 'raft'), args.of_model_path_bwd)
        else:
            self.of_model_bwd = self.of_model # Fallback to the same model

        self.of_scale = args.of_scale
        self.use_bidirectional_warp = not getattr(args, 'disable_bidirectional_warp', False)
        # Occlusion detection parameters
        self.photometric_loss = nn.L1Loss(reduction='none')
        self.occlusion_threshold = getattr(args, 'occlusion_threshold', 0.5)
        self.flow_consistency_alpha = getattr(args, 'flow_consistency_alpha', 0.01)
        self.fusion_confidence_threshold = getattr(args, 'fusion_confidence_threshold', 0.1)

    def load_optical_flow_model(self, args, model_name='raft', model_path=None):
        """Loads an optical flow model (UNCHANGED)."""
        if model_path and model_path.lower() != "raft":
            model = ptlflow.get_model(model_name)
            ckpt = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if 'state_dict' in ckpt: state_dict = ckpt['state_dict']
            elif 'model' in ckpt: state_dict = ckpt['model']
            else: state_dict = ckpt
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
        elif model_name.lower() == 'raft':
            model = torch.nn.DataParallel(RAFT(args))
            if model_path:
                model.load_state_dict(torch.load(model_path))
            model = model.module
        else:
            model = ptlflow.get_model(model_name)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    # --- Other helper methods (unchanged) ---
    def cvt_ts2np(self, t):
        t = t.detach()
        n = t.squeeze().permute((1, 2, 0)).cpu().numpy()
        return n

    def enhance_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None: m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def denoise_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None: m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input, img_path=None):
        eps = 1e-4
        input = input + eps

        L11, L12 = pair_downsampler(input)
        L_pred1 = L11 - self.denoise_1(L11)
        L_pred2 = L12 -self.denoise_1(L12)
        L2 = input - self.denoise_1(input)
        L2 = torch.clamp(L2, eps, 1)

        if self.is_new_seq:
            self.last_H3_wp = torch.zeros_like(L2)
            self.last_s3_wp = torch.zeros_like(L2)
            self.last_H31_wp = torch.zeros_like(L11)
            self.last_H32_wp = torch.zeros_like(L11)
            self.last_s31_wp = torch.zeros_like(L11)
            self.last_s32_wp = torch.zeros_like(L11)
        else:
            self.last_H3_wp, self.last_s3_wp = self.update_cache(self.last_H3, self.last_s3, L2.detach(), img_path)
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

    def _loss(self, input, img_path=None):
        outputs = self(input, img_path)
        # The forward pass returns 23 values, but the loss function only expects 21 of them.
        # We must exclude the last two (H3_denoised1, H3_denoised2)
        loss = self._criterion(input, *outputs[:-2])
        self.update_H3(outputs[13], outputs[14]) # H3 and s3
        return loss

    def update_H3(self, H3, s3):
        self.last_H3 = H3.detach()
        self.last_s3 = s3.detach()

    # --- NEW HELPER METHODS FOR BIDIRECTIONAL FLOW ---
    def _load_frame(self, path):
        """Loads a frame from a path and converts it to a tensor."""
        image = Image.open(path).convert('RGB')
        return to_tensor(image).unsqueeze(0).to(self.last_H3.device)

    def _compute_single_flow(self, img1, img2, model):
        """Computes flow from img1->img2 using the provided model."""
        with torch.no_grad():
            ht_org, wd_org = img1.shape[-2:]
            ht, wd = ht_org // self.of_scale, wd_org // self.of_scale

            img1_scaled = F.interpolate(img1, (ht, wd), mode='bilinear', align_corners=False)
            img2_scaled = F.interpolate(img2, (ht, wd), mode='bilinear', align_corners=False)

            img1_flow = equalize((img1_scaled * 255).to(torch.uint8)).float()
            img2_flow = equalize((img2_scaled * 255).to(torch.uint8)).float()

            model = model.to(img1.device)

            flow = None
            if isinstance(model, RAFT):
                padder = InputPadder(img1_flow.shape[-2:])
                img1_padded, img2_padded = padder.pad(img1_flow, img2_flow)
                _, flow = model(img1_padded, img2_padded, iters=20, test_mode=True)
                flow = padder.unpad(flow)
            else: # ptlflow model
                inputs = {'images': torch.stack([img1_flow, img2_flow], dim=1)}
                try:
                    outputs = model(inputs)
                    flow = outputs['flows'][:, 0]
                except KeyError:
                    flow = torch.zeros(img1_flow.shape[0], 2, img1_flow.shape[2], img1_flow.shape[3], device=img1_flow.device)

            flow_up = F.interpolate(flow, (ht_org, wd_org), mode='bilinear', align_corners=False)
            flow_up[:, 0] *= wd_org / flow.shape[3]
            flow_up[:, 1] *= ht_org / flow.shape[2]
            return flow_up

    def _compute_bidirectional_flow(self, img1, img2, model):
        """Computes flow img1->img2 and img2->img1 using the same model."""
        flow_fwd = self._compute_single_flow(img1, img2, model)
        flow_bwd = self._compute_single_flow(img2, img1, model)
        return flow_fwd, flow_bwd

    def _get_occlusion_mask(self, flow_fwd, flow_bwd):
        """Calculates an occlusion mask based on forward-backward consistency."""
        warped_flow_bwd, _ = warp_tensor(flow_fwd, flow_bwd, flow_bwd)
        
        flow_diff = flow_fwd + warped_flow_bwd
        consistency_error = torch.norm(flow_diff, dim=1, keepdim=True)
        
        flow_magnitude = torch.norm(flow_fwd, dim=1, keepdim=True)
        adaptive_threshold = self.occlusion_threshold + self.flow_consistency_alpha * flow_magnitude
        
        occlusion_mask = (consistency_error > adaptive_threshold).float()
        return occlusion_mask

    def update_cache(self, last_H3, last_s3, L2, img_path):
        """
        Updates cache with consistency-checked, intelligently-fused bidirectional warping.
        """
        # Forward warp (t-1 -> t)
        flow_fwd, flow_fwd_bwd = self._compute_bidirectional_flow(last_H3, L2, self.of_model)
        mask_fwd = self._get_occlusion_mask(flow_fwd, flow_fwd_bwd)
        warped_H3_fwd, _ = warp_tensor(flow_fwd, last_H3, L2)
        warped_s3_fwd, _ = warp_tensor(flow_fwd, last_s3, L2)

        # Backward warp (t+1 -> t) if possible
        if self.use_bidirectional_warp:
            next_frame_path = get_next_frame_path(img_path)
            if next_frame_path:
                try:
                    L2_next = self._load_frame(next_frame_path)
                    eps = 1e-4
                    L2_next_processed = L2_next + eps
                    L2_next_denoised = L2_next_processed - self.denoise_1(L2_next_processed)
                    L2_next = torch.clamp(L2_next_denoised, eps, 1)

                    flow_bwd = self._compute_single_flow(L2_next, L2, self.of_model_bwd)
                    flow_bwd_fwd = self._compute_single_flow(L2, L2_next, self.of_model_bwd)
                    
                    mask_bwd = self._get_occlusion_mask(flow_bwd, flow_bwd_fwd)
                    warped_H3_bwd, _ = warp_tensor(flow_bwd, L2_next, L2)

                    confidence_fwd = 1.0 - mask_fwd
                    confidence_bwd = 1.0 - mask_bwd
                    total_confidence = confidence_fwd + confidence_bwd + 1e-8 # Avoid division by zero
                    
                    w_fwd = confidence_fwd / total_confidence
                    w_bwd = confidence_bwd / total_confidence

                    blended_H3 = w_fwd * warped_H3_fwd + w_bwd * warped_H3_bwd

                    is_occluded = (confidence_fwd + confidence_bwd < self.fusion_confidence_threshold).float()
                    final_H3 = blended_H3 * (1 - is_occluded) + L2 * is_occluded
                    final_s3 = warped_s3_fwd

                    return final_H3, final_s3
                except Exception:
                    pass

        # Fallback to forward-only warp
        final_H3 = warped_H3_fwd
        final_s3 = warped_s3_fwd
        
        return final_H3, final_s3


class Finetunemodel(nn.Module):
    def __init__(self, args):
        super(Finetunemodel, self).__init__()
        self.args = args

        self.enhance = Enhancer(layers=3, channels=64)
        self.denoise_1 = Denoise_1(chan_embed=48)
        self.denoise_2 = Denoise_2(chan_embed=48)

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

        self.of_model = self.load_optical_flow_model(args, getattr(args, 'of_model_name', 'raft'), getattr(args, 'of_model_path', None))
        # Load a second OF model for the backward pass if specified
        if hasattr(args, 'of_model_path_bwd') and args.of_model_path_bwd:
            self.of_model_bwd = self.load_optical_flow_model(args, getattr(args, 'of_model_name_bwd', 'raft'), args.of_model_path_bwd)
        else:
            self.of_model_bwd = self.of_model # Fallback to the same model

        self.of_scale = args.of_scale
        self.use_bidirectional_warp = not getattr(args, 'disable_bidirectional_warp', False)
        self.photometric_loss = nn.L1Loss(reduction='none')
        self.occlusion_threshold = getattr(args, 'occlusion_threshold', 0.5)
        self.flow_consistency_alpha = getattr(args, 'flow_consistency_alpha', 0.01)
        self.fusion_confidence_threshold = getattr(args, 'fusion_confidence_threshold', 0.1)

    def load_optical_flow_model(self, args, model_name='raft', model_path=None):
        """Loads an optical flow model (UNCHANGED)."""
        if model_path is None:
            model = ptlflow.get_model(model_name)
        elif model_path.lower() != "raft":
            model = ptlflow.get_model(model_name)
            ckpt = torch.load(model_path, map_location='cpu')
            if 'state_dict' in ckpt: state_dict = ckpt['state_dict']
            elif 'model' in ckpt: state_dict = ckpt['model']
            else: state_dict = ckpt
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
        elif model_name.lower() == 'raft':
            model = torch.nn.DataParallel(RAFT(args))
            if model_path:
                model.load_state_dict(torch.load(model_path))
            model = model.module
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

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

    def forward(self, input, img_path=None):
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
            self.last_H3_wp, self.last_s3_wp = self.update_cache(self.last_H3, self.last_s3, L2.detach(), img_path)

        s2 = self.enhance(torch.cat([self.last_H3_wp, self.last_s3_wp, L2], 1).detach())
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)

        # if self.is_new_seq:
        #     self.last_H3_wp = H2.detach()
        #     self.last_s3_wp = H2.detach()

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
        
    # --- COPY THE IDENTICAL HELPER AND UPDATE_CACHE METHODS HERE ---
    def _load_frame(self, path):
        """Loads a frame from a path and converts it to a tensor."""
        image = Image.open(path).convert('RGB')
        return to_tensor(image).unsqueeze(0).to(self.last_H3.device)

    def _compute_single_flow(self, img1, img2, model):
        """Computes flow from img1->img2 using the provided model."""
        with torch.no_grad():
            ht_org, wd_org = img1.shape[-2:]
            ht, wd = ht_org // self.of_scale, wd_org // self.of_scale

            img1_scaled = F.interpolate(img1, (ht, wd), mode='bilinear', align_corners=False)
            img2_scaled = F.interpolate(img2, (ht, wd), mode='bilinear', align_corners=False)

            img1_flow = equalize((img1_scaled * 255).to(torch.uint8)).float()
            img2_flow = equalize((img2_scaled * 255).to(torch.uint8)).float()

            model = model.to(img1.device)

            flow = None
            if isinstance(model, RAFT):
                padder = InputPadder(img1_flow.shape[-2:])
                img1_padded, img2_padded = padder.pad(img1_flow, img2_flow)
                _, flow = model(img1_padded, img2_padded, iters=20, test_mode=True)
                flow = padder.unpad(flow)
            else: # ptlflow model
                inputs = {'images': torch.stack([img1_flow, img2_flow], dim=1)}
                try:
                    outputs = model(inputs)
                    flow = outputs['flows'][:, 0]
                except KeyError:
                    flow = torch.zeros(img1_flow.shape[0], 2, img1_flow.shape[2], img1_flow.shape[3], device=img1_flow.device)

            flow_up = F.interpolate(flow, (ht_org, wd_org), mode='bilinear', align_corners=False)
            flow_up[:, 0] *= wd_org / flow.shape[3]
            flow_up[:, 1] *= ht_org / flow.shape[2]
            return flow_up

    def _compute_bidirectional_flow(self, img1, img2, model):
        """Computes flow img1->img2 and img2->img1 using the same model."""
        flow_fwd = self._compute_single_flow(img1, img2, model)
        flow_bwd = self._compute_single_flow(img2, img1, model)
        return flow_fwd, flow_bwd

    def _get_occlusion_mask(self, flow_fwd, flow_bwd):
        """Calculates an occlusion mask based on forward-backward consistency."""
        warped_flow_bwd, _ = warp_tensor(flow_fwd, flow_bwd, flow_bwd)
        
        flow_diff = flow_fwd + warped_flow_bwd
        consistency_error = torch.norm(flow_diff, dim=1, keepdim=True)
        
        flow_magnitude = torch.norm(flow_fwd, dim=1, keepdim=True)
        adaptive_threshold = self.occlusion_threshold + self.flow_consistency_alpha * flow_magnitude
        
        occlusion_mask = (consistency_error > adaptive_threshold).float()
        return occlusion_mask
    
    def update_cache(self, last_H3, last_s3, L2, img_path):
        """
        Updates cache with consistency-checked, intelligently-fused bidirectional warping.
        """
        # 1. Forward warp (t-1 -> t)
        flow_fwd, flow_fwd_bwd = self._compute_bidirectional_flow(last_H3, L2, self.of_model)
        mask_fwd = self._get_occlusion_mask(flow_fwd, flow_fwd_bwd)
        warped_H3_fwd, _ = warp_tensor(flow_fwd, last_H3, L2)
        warped_s3_fwd, _ = warp_tensor(flow_fwd, last_s3, L2)

        # 2. Backward warp (t+1 -> t) if possible
        if self.use_bidirectional_warp:
            next_frame_path = get_next_frame_path(img_path)
            if next_frame_path:
                try:
                    L2_next_raw = self._load_frame(next_frame_path)

                    eps = 1e-4
                    L2_next_processed = L2_next_raw + eps
                    L2_next_denoised = L2_next_processed - self.denoise_1(L2_next_processed)
                    L2_next_denoised = torch.clamp(L2_next_denoised, eps, 1)


                    flow_bwd = self._compute_single_flow(L2_next_denoised, L2, self.of_model_bwd)
                    flow_bwd_fwd = self._compute_single_flow(L2, L2_next_denoised, self.of_model_bwd)

                    mask_bwd = self._get_occlusion_mask(flow_bwd, flow_bwd_fwd)
                    warped_H3_bwd, _ = warp_tensor(flow_bwd, L2_next_raw, L2)

                    # 3. Intelligent Fusion
                    confidence_fwd = 1.0 - mask_fwd
                    confidence_bwd = 1.0 - mask_bwd
                    total_confidence = confidence_fwd + confidence_bwd + 1e-8 # Avoid division by zero
                    
                    w_fwd = confidence_fwd / total_confidence
                    w_bwd = confidence_bwd / total_confidence

                    blended_H3 = w_fwd * warped_H3_fwd + w_bwd * warped_H3_bwd

                    is_occluded = (confidence_fwd + confidence_bwd < self.fusion_confidence_threshold).float()
                    
                    final_H3 = blended_H3 * (1 - is_occluded) + warped_H3_fwd * is_occluded
                    final_s3 = warped_s3_fwd

                    return final_H3, final_s3
                except Exception:
                    pass

        # Fallback to forward-only warp
        final_H3 = warped_H3_fwd
        final_s3 = warped_s3_fwd
        
        return final_H3, final_s3
