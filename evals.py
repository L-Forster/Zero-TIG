import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import argparse
import logging
from pathlib import Path
import re
from collections import defaultdict

# Attempt to import cv2, warn if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    # No need to log warning here, will be logged by setup_logging if it's the first import
    pass


def setup_logging(save_dir):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    # Remove existing handlers to avoid duplicate logs if re-running in same session
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    
    # File handler
    log_file = os.path.join(save_dir, 'evaluation_log.txt')
    file_handler = logging.FileHandler(log_file, mode='w') # Overwrite log file each run
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    
    # Initial warning for cv2 if not available
    if not CV2_AVAILABLE:
        logging.warning("OpenCV (cv2) is not installed. SSIM calculation might use Pillow for grayscale if cv2 is not found.")
    return logging.getLogger()


class ImageQualityEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])
        logging.info(f"ImageQualityEvaluator initialized on device: {self.device}")

    def load_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            return np.array(img)
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None

    def calculate_psnr(self, img1, img2):
        try:
            img1 = np.clip(img1, 0, 255).astype(np.uint8)
            img2 = np.clip(img2, 0, 255).astype(np.uint8)
            return peak_signal_noise_ratio(img1, img2, data_range=255)
        except Exception as e:
            logging.error(f"Error calculating PSNR: {e}")
            return np.nan

    def calculate_ssim(self, img1, img2):
        try:
            img1_u8 = np.clip(img1, 0, 255).astype(np.uint8)
            img2_u8 = np.clip(img2, 0, 255).astype(np.uint8)

            if img1_u8.ndim == 3 and img1_u8.shape[2] == 3: # Check if color
                if CV2_AVAILABLE:
                    img1_gray = cv2.cvtColor(img1_u8, cv2.COLOR_RGB2GRAY)
                    img2_gray = cv2.cvtColor(img2_u8, cv2.COLOR_RGB2GRAY)
                else: # Fallback to Pillow if cv2 not available
                    img1_gray = np.array(Image.fromarray(img1_u8).convert('L'))
                    img2_gray = np.array(Image.fromarray(img2_u8).convert('L'))
            else: # Already grayscale or unexpected 1-channel format
                img1_gray = img1_u8
                img2_gray = img2_u8
                if img1_u8.ndim == 3 or img2_u8.ndim == 3: # Should not happen if previous check is correct
                    logging.warning("SSIM: Mixed color/grayscale formats despite checks.")
            
            min_dim = min(img1_gray.shape[0], img1_gray.shape[1])
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            
            if win_size < 3 : # skimage SSIM default is 7, requires min_dim >= win_size
                logging.warning(f"SSIM window size ({win_size}) is too small for image dimensions ({img1_gray.shape}). SSIM may be unreliable or error. Setting to None.")
                return np.nan # Or handle as per requirement, e.g., skip SSIM.
            
            return structural_similarity(img1_gray, img2_gray, data_range=255, win_size=win_size)
        except Exception as e:
            logging.error(f"Error calculating SSIM: {e}")
            return np.nan

    def calculate_lpips(self, img1, img2):
        try:
            img1_pil = Image.fromarray(np.clip(img1, 0, 255).astype(np.uint8))
            img2_pil = Image.fromarray(np.clip(img2, 0, 255).astype(np.uint8))
            img1_tensor = self.transform(img1_pil).unsqueeze(0).to(self.device) * 2.0 - 1.0
            img2_tensor = self.transform(img2_pil).unsqueeze(0).to(self.device) * 2.0 - 1.0
            with torch.no_grad():
                return self.lpips_model(img1_tensor, img2_tensor).item()
        except Exception as e:
            logging.error(f"Error calculating LPIPS: {e}")
            return np.nan

    def evaluate_image_pair(self, pred_path, gt_path):
        pred_img = self.load_image(pred_path)
        gt_img = self.load_image(gt_path)
        if pred_img is None or gt_img is None:
            return None
        if pred_img.shape != gt_img.shape:
            logging.warning(f"Shape mismatch: Pred {pred_path} {pred_img.shape}, GT {gt_path} {gt_img.shape}. Resizing pred.")
            try:
                pred_img = np.array(Image.fromarray(pred_img).resize((gt_img.shape[1], gt_img.shape[0]), Image.Resampling.LANCZOS))
            except Exception as e:
                logging.error(f"Error resizing {pred_path}: {e}")
                return None
        return {
            'psnr': self.calculate_psnr(pred_img, gt_img),
            'ssim': self.calculate_ssim(pred_img, gt_img),
            'lpips': self.calculate_lpips(pred_img, gt_img)
        }

def extract_number_from_filename(filename):
    match = re.search(r'(\d+)', os.path.splitext(os.path.basename(filename))[0])
    return int(match.group(1)) if match else 0

def sort_image_files(file_list):
    return sorted(file_list, key=extract_number_from_filename)

def get_gt_frames_from_test_list(gt_base_dir, test_list_file_path):
    gt_frames_info = []
    if not os.path.exists(test_list_file_path):
        logging.error(f"Test list file not found: {test_list_file_path}")
        return gt_frames_info

    with open(test_list_file_path, 'r') as f:
        sequence_folders = [line.strip() for line in f if line.strip()]

    for seq_folder_name in sequence_folders:
        for brightness_val in ["10", "20"]:
            gt_brightness_folder = f"normal_light_{brightness_val}"
            current_gt_seq_brightness_dir = os.path.join(gt_base_dir, seq_folder_name, gt_brightness_folder)
            
            if not os.path.isdir(current_gt_seq_brightness_dir):
                logging.warning(f"GT dir for seq {seq_folder_name}, brightness {gt_brightness_folder} not found: {current_gt_seq_brightness_dir}")
                continue

            gt_image_paths_for_level = sort_image_files(glob.glob(os.path.join(current_gt_seq_brightness_dir, "*.png")))
            if not gt_image_paths_for_level:
                 gt_image_paths_for_level = sort_image_files(glob.glob(os.path.join(current_gt_seq_brightness_dir, "*.jpg"))) # Try jpg if no png

            for gt_path in gt_image_paths_for_level:
                frame_number_str = os.path.splitext(os.path.basename(gt_path))[0]
                frame_number_str = re.sub(r'^[a-zA-Z_]*', '', frame_number_str) # Remove common prefixes like 'frame_'

                gt_frames_info.append({
                    "sequence_name": seq_folder_name,
                    "brightness_level_val": brightness_val,
                    "frame_number_str": frame_number_str,
                    "gt_path": gt_path
                })
    logging.info(f"Collected {len(gt_frames_info)} ground truth frame references from test_list.txt.")
    return gt_frames_info


def main():
    parser = argparse.ArgumentParser(description='Evaluate image enhancement results per epoch based on test_list.txt ordering')
    parser.add_argument('--results_exp_dir', type=str, default='./EXP/Train-20250606-161137/',
                        help='Base directory of the experiment (e.g., ./EXP/Train-20250606-161137)')

    parser.add_argument('--gt_base_dir', type=str, default='./lowlight_dataset/gt/',
                        help='Base directory for ALL ground truth sequences (e.g., /lowlight_dataset/gt/)')
                        
    parser.add_argument('--test_list_file', type=str, default='./lowlight_dataset/test_list.txt',
                        help='Path to the test_list.txt file (e.g. /lowlight_dataset/test_list.txt)')

    parser.add_argument('--max_epochs', type=int, default=4, # Default added
                        help='Maximum epoch number to evaluate (e.g., if epochs are 0-4, pass 4, meaning 5 epochs total)')

    parser.add_argument('--image_type_to_eval', type=str, default='both', choices=['denoise', 'enhance', 'both'],
                        help='Type of prediction images to evaluate from the results folder')

    parser.add_argument('--save_dir_base', type=str, default='./evaluation_ordered_per_epoch/',
                        help='Base directory to save evaluation results. A subfolder will be created.')

    args = parser.parse_args()

    experiment_name = os.path.basename(args.results_exp_dir.rstrip('/'))
    args.save_dir = os.path.join(args.save_dir_base, experiment_name if experiment_name else "eval_run")
    os.makedirs(args.save_dir, exist_ok=True)

    logger = setup_logging(args.save_dir)
    logger.info(f"Starting ORDERED evaluation with args: {args}")

    evaluator = ImageQualityEvaluator()
    ordered_gt_frames = get_gt_frames_from_test_list(args.gt_base_dir, args.test_list_file)
    if not ordered_gt_frames:
        logger.error("No ground truth frames mapped from test_list.txt. Aborting.")
        return

    all_evaluation_results = []
    epochs_to_evaluate = range(args.max_epochs + 1)

    pred_types_to_process = []
    if args.image_type_to_eval == 'enhance' or args.image_type_to_eval == 'both':
        pred_types_to_process.append('enhance')
    if args.image_type_to_eval == 'denoise' or args.image_type_to_eval == 'both':
        pred_types_to_process.append('denoise')

    for epoch_num in epochs_to_evaluate:
        logger.info(f"--- Evaluating Epoch {epoch_num} ---")
        processed_for_epoch = 0
        for gt_info in ordered_gt_frames:
            seq_name = gt_info["sequence_name"]
            brightness_val = gt_info["brightness_level_val"]
            frame_num_str = gt_info["frame_number_str"]
            gt_full_path = gt_info["gt_path"]
            pred_brightness_prefix = f"low_light_{brightness_val}"
            
            for pred_img_type in pred_types_to_process:
                expected_pred_filename = f"{pred_brightness_prefix}_{frame_num_str}_{pred_img_type}_{epoch_num}.png"
                pred_full_path = os.path.join(args.results_exp_dir, 'result', pred_img_type, expected_pred_filename)

                if not os.path.exists(pred_full_path):
                    logging.debug(f"Pred NOT FOUND for E{epoch_num}, GT_Base: {os.path.basename(gt_full_path)}, Expected: {pred_full_path}")
                    continue
                
                metrics = evaluator.evaluate_image_pair(pred_full_path, gt_full_path)
                if metrics:
                    all_evaluation_results.append({
                        'sequence_name': seq_name,
                        'epoch': epoch_num,
                        'prediction_type': pred_img_type,
                        'brightness_condition_val': brightness_val,
                        'frame_number': frame_num_str,
                        'pred_path': pred_full_path,
                        'gt_path': gt_full_path,
                        **metrics
                    })
                    processed_for_epoch +=1
                else:
                    logging.warning(f"Metric calculation failed for Pred: {pred_full_path}, GT: {gt_full_path}")
        logger.info(f"--- Epoch {epoch_num}: Processed {processed_for_epoch} prediction files successfully. ---")


    if not all_evaluation_results:
        logger.error("No results were generated. Check paths, file structures, and epoch range.")
        return

    df_results = pd.DataFrame(all_evaluation_results)
    
    # 1. Detailed results per file (already good)
    detailed_csv_path = os.path.join(args.save_dir, 'detailed_ordered_results_per_file.csv') # Renamed for clarity
    df_results.to_csv(detailed_csv_path, index=False)
    logger.info(f"Detailed results per file saved to {detailed_csv_path}")

    # 2. Granular summary (already good)
    granular_summary_stats = df_results.groupby(['sequence_name', 'epoch', 'prediction_type', 'brightness_condition_val']).agg(
        num_images=('frame_number', 'count'),
        psnr_mean=('psnr', 'mean'), psnr_std=('psnr', 'std'),
        ssim_mean=('ssim', 'mean'), ssim_std=('ssim', 'std'),
        lpips_mean=('lpips', 'mean'), lpips_std=('lpips', 'std')
    ).reset_index()
    granular_summary_csv_path = os.path.join(args.save_dir, 'summary_results_granular_per_epoch.csv') # Renamed
    granular_summary_stats.to_csv(granular_summary_csv_path, index=False)
    logger.info(f"Granular summary results saved to {granular_summary_csv_path}")

    # 3. NEW: Overall average for each epoch, split by prediction_type
    epoch_level_summary = df_results.groupby(['epoch', 'prediction_type']).agg(
        total_images=('frame_number', 'count'),
        psnr_mean=('psnr', 'mean'), psnr_std=('psnr', 'std'),
        ssim_mean=('ssim', 'mean'), ssim_std=('ssim', 'std'),
        lpips_mean=('lpips', 'mean'), lpips_std=('lpips', 'std')
    ).reset_index()
    epoch_level_summary_csv_path = os.path.join(args.save_dir, 'summary_overall_per_epoch_type.csv')
    epoch_level_summary.to_csv(epoch_level_summary_csv_path, index=False)
    logger.info(f"Overall summary per epoch and type saved to {epoch_level_summary_csv_path}")


    logger.info("\n--- Granular Summary (Per Sequence, Epoch, Type, Brightness) ---")
    for _, row in granular_summary_stats.iterrows():
        logger.info(
            f"Seq: {row['sequence_name']}, Epoch: {row['epoch']}, Type: {row['prediction_type']}, "
            f"Brightness: {row['brightness_condition_val']}, Images: {row['num_images']}\n"
            f"  PSNR: {row['psnr_mean']:.4f} ± {row['psnr_std']:.4f}\n"
            f"  SSIM: {row['ssim_mean']:.4f} ± {row['ssim_std']:.4f}\n"
            f"  LPIPS: {row['lpips_mean']:.4f} ± {row['lpips_std']:.4f}\n"
        )
        
    logger.info("\n--- Overall Summary (Per Epoch, Type) ---")
    for _, row in epoch_level_summary.iterrows():
        logger.info(
            f"Epoch: {row['epoch']}, Prediction Type: {row['prediction_type']}, Total Images: {row['total_images']}\n"
            f"  Avg PSNR: {row['psnr_mean']:.4f} ± {row['psnr_std']:.4f}\n"
            f"  Avg SSIM: {row['ssim_mean']:.4f} ± {row['ssim_std']:.4f}\n"
            f"  Avg LPIPS: {row['lpips_mean']:.4f} ± {row['lpips_std']:.4f}\n"
        )

    logger.info("Ordered evaluation completed.")

if __name__ == '__main__':
    main()