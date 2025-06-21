import sys
import time
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

sys.path.append('core')

# Import models
from model.RAFT.raft import RAFT
import ptlflow
from utils import flow_viz
from utils.utils import InputPadder
from ptlflow.utils.io_adapter import IOAdapter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_raft_model(checkpoint_path):
    """Load RAFT model from checkpoint."""
    args = argparse.Namespace()
    args.small = False
    args.alternate_corr = False
    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.module
    model.to(DEVICE)
    model.eval()
    return model

def load_dpflow_model():
    """Load DPFlow model using ptlflow."""
    model = ptlflow.get_model('dpflow')
    model.to(DEVICE)
    model.eval()
    return model

def load_image(imfile):
    """Load and preprocess image for optical flow."""
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = cv2.resize(img, [640, 360])  # Standard size for flow computation
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def save_flow_visualization(flow, output_path):
    """Save optical flow as color-coded visualization."""
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
    flow_color = flow_viz.flow_to_image(flow_np)
    # Increase brightness by 50%
    flow_color = cv2.convertScaleAbs(flow_color, alpha=2.0, beta=0)
    
    # Convert RGB to BGR for OpenCV
    cv2.imwrite(output_path, flow_color[:, :, [2, 1, 0]])

def save_warped_image(image, flow, output_path):
    """Warp image with flow and save it."""
    image_np = image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()

    h, w = flow_np.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    map1 = (xx + flow_np[..., 0]).astype(np.float32)
    map2 = (yy + flow_np[..., 1]).astype(np.float32)

    warped_image = cv2.remap(image_np, map1, map2, cv2.INTER_LINEAR)
    
    # Increase brightness by 50%
    warped_image = cv2.convertScaleAbs(warped_image, alpha=2.0, beta=0)

    # Convert RGB to BGR for OpenCV
    cv2.imwrite(output_path, warped_image[:, :, ::-1])

def compute_flow_raft(model, image1_path, image2_path):
    """Compute optical flow using RAFT."""
    image1 = np.array(Image.open(image1_path)).astype(np.uint8)
    image2 = np.array(Image.open(image2_path)).astype(np.uint8)

    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()[None].to(DEVICE)
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()[None].to(DEVICE)

    padder = InputPadder(image1.shape)
    image1_pad, image2_pad = padder.pad(image1, image2)
    
    with torch.no_grad():
        _, flow = model(image1_pad, image2_pad, iters=20, test_mode=True)
    
    return padder.unpad(flow)

def compute_flow_dpflow(model, image1_path, image2_path):
    """Compute optical flow using DPFlow."""
    # Load images
    image1 = np.array(Image.open(image1_path)).astype(np.uint8)
    image2 = np.array(Image.open(image2_path)).astype(np.uint8)
    
    # The IOAdapter expects a list of numpy images.
    # It will handle resizing, normalization, and conversion to tensor.
    io_adapter = IOAdapter(model, image1.shape[:2])
    inputs = io_adapter.prepare_inputs([image1, image2])

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(DEVICE)

    with torch.no_grad():
        predictions = model(inputs)
        # The output of ptlflow models is a dict with a 'flows' tensor of shape BxNx2xHxW.
        # N is the number of predictions. We only need one for inference.
        # Squeeze the N dimension to get Bx2xHxW, which is the format expected by other functions.
        flow = predictions['flows'].squeeze(1)
    
    return flow

def create_zero_flow(image_shape):
    """Create zero optical flow (for 'nothing' comparison)."""
    _, _, h, w = image_shape
    return torch.zeros(1, 2, h, w, device=DEVICE)

def main():
    parser = argparse.ArgumentParser(description="Compare optical flow models on lowlight dataset")
    parser.add_argument('--input_dir', type=str, default='./lowlight_dataset/input/S11_gift_wrap/low_light_10',
                        help='Input directory with image sequences')
    parser.add_argument('--output_dir', type=str, default='./flow_comparison',
                        help='Output directory for flow visualizations')
    parser.add_argument('--raft_weights', type=str, default='./weights/raft-sintel.pth',
                        help='Path to RAFT model weights')
    parser.add_argument('--models', type=str, nargs='+', default=['raft', 'dpflow', 'none'],
                        choices=['raft', 'dpflow', 'none'],
                        help='Models to compare')
    parser.add_argument('--viz_mode', type=str, default='warp', choices=['flow', 'warp'], 
                        help='Visualization mode: flow heatmap or warped image')
    
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    for model_name in args.models:
        os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
    
    # Load models
    models = {}
    if 'raft' in args.models:
        print("Loading RAFT model...")
        models['raft'] = load_raft_model(args.raft_weights)
    
    if 'dpflow' in args.models:
        print("Loading DPFlow model...")
        models['dpflow'] = load_dpflow_model()
    
    # Find image pairs
    image_files = sorted(glob.glob(os.path.join(args.input_dir, '*.png')) + 
                        glob.glob(os.path.join(args.input_dir, '*.jpg')))
    
    if len(image_files) < 2:
        print(f"Not enough images found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Process image pairs
    for i, (img1_path, img2_path) in enumerate(zip(image_files[:-1], image_files[1:])):
        img1_name = os.path.splitext(os.path.basename(img1_path))[0]
        img2_name = os.path.splitext(os.path.basename(img2_path))[0]
        
        print(f"Processing pair {i+1}: {img1_name} -> {img2_name}")
        
        # Compute flow for each model
        for model_name in args.models:
            output_path = os.path.join(args.output_dir, model_name, 
                                     f"flow_{img1_name}_to_{img2_name}.png")
            
            if model_name == 'raft':
                flow = compute_flow_raft(models['raft'], img1_path, img2_path)
            elif model_name == 'dpflow':
                flow = compute_flow_dpflow(models['dpflow'], img1_path, img2_path)
            elif model_name == 'none':
                # To get the shape, we need to load the image.
                h, w, _ = np.array(Image.open(img1_path)).shape
                flow = create_zero_flow((1, 3, h, w))
            
            if args.viz_mode == 'flow':
                save_flow_visualization(flow, output_path)
            elif args.viz_mode == 'warp':
                # For warping, we need to load the second image again
                image2_for_warp = torch.from_numpy(np.array(Image.open(img2_path)).astype(np.uint8)).permute(2, 0, 1).float()[None].to(DEVICE)
                save_warped_image(image2_for_warp, flow, output_path)

            print(f"  Saved {model_name} flow to {output_path}")
    
    print(f"\nComparison complete! Check results in {args.output_dir}")

if __name__ == '__main__':
    main()
