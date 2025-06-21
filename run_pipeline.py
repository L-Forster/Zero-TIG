import os
import sys
import glob
import logging
import argparse
import subprocess
import pandas as pd
import json
import time

def get_dataset_type(dataset_dir_name):
    """Map directory names to dataset types expected by the scripts"""
    mapping = {
        'lowlight_dataset': 'lowlight_dataset',  # Use lowlight_dataset as the dataset type
        'RLV': 'RLV',
        'BVI-RLV': 'RLV', 
        'DID_1080': 'DID',
        'SDSD-indoor': 'SDSD',  # Use SDSD loader for indoor
        'SDSD-outdoor': 'SDSD',  # Use SDSD loader for outdoor
        '3_SDSD': 'SDSD',  # Handle the actual SDSD directory name
    }
    return mapping.get(dataset_dir_name, dataset_dir_name)

def setup_logging(log_file):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    # Remove existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO, format=log_format,
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler(sys.stdout)])
    return logging.getLogger()

def find_latest_run_dir(base_dir):
    """Finds the most recently created 'Train-*' directory."""
    list_of_dirs = glob.glob(os.path.join(base_dir, 'Train-*'))
    if not list_of_dirs:
        return None
    return max(list_of_dirs, key=os.path.getctime)

def find_latest_weights_file(model_epochs_dir):
    """Finds the most recent weights file in the model_epochs directory."""
    weights_files = glob.glob(os.path.join(model_epochs_dir, 'weights_*.pt'))
    if not weights_files:
        return None
    
    # Extract epoch number from filename and find the max
    latest_file = max(weights_files, key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    return latest_file

def run_command(command, logger):
    """Executes a command and logs its output."""
    logger.info(f"Executing command: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Log output line by line in real-time
        for line in iter(process.stdout.readline, ''):
            logger.info(line.strip())
            
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"Command failed with return code {return_code}")
            return False
    except Exception as e:
        logger.error(f"Failed to execute command: {' '.join(command)}")
        logger.error(f"Error: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the complete Zero-TIG training, prediction, and evaluation pipeline.")
    parser.add_argument('--datasets', nargs='+', required=True,
                        help="List of dataset names to process (e.g., RLV LOL-v1). Each name must correspond to a folder in the base_data_dir.")
    parser.add_argument('--base_data_dir', type=str, default='./data/',
                        help="The base directory that CONTAINS all individual dataset folders (e.g., ./data/RLV, ./data/AnotherDataset).")
    parser.add_argument('--weights_dir', type=str, default='./weights/',
                        help="Directory containing pre-trained model weights.")
    parser.add_argument('--pretrain_weights_file', type=str, default='BVI-RLV.pt',
                        help="Filename of the pre-trained weights to use for fine-tuning.")
    parser.add_argument('--base_exp_dir', type=str, default='./PIPELINE_EXP',
                        help="Base directory to save all outputs from the pipeline.")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of workers for data loading.")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of epochs for fine-tuning.")
    parser.add_argument('--eval_name', type=str, default='run',
                        help="Name for the evaluation run, used for log and metric file names.")
    parser.add_argument('--of_model_path', type=str, default=None,
                        help="Path to a custom optical flow model checkpoint.")
    parser.add_argument('--of_model_name', type=str, default='dpflow',
                        help="Name of the optical flow model to use (e.g., 'raft', 'dpflow').")
    parser.add_argument('--disable_self_ensemble', action='store_true',
                        help="If set, disables the self-ensemble module in the model.")
    
    args = parser.parse_args()

    os.makedirs(args.base_exp_dir, exist_ok=True)
    logger = setup_logging(os.path.join(args.base_exp_dir, 'pipeline_log.txt'))

    logger.info(f"Starting pipeline with arguments: {args}")
    start_time = time.time()

    for dataset_name in args.datasets:
        logger.info(f"========== PROCESSING DATASET: {dataset_name} ==========")

        # --- Setup paths for the current dataset ---
        train_base_dir = os.path.join(args.base_exp_dir, dataset_name, 'training')
        predict_save_dir = os.path.join(args.base_exp_dir, dataset_name, 'predictions')
        eval_save_dir = os.path.join(args.base_exp_dir, dataset_name, 'evaluation')
        os.makedirs(train_base_dir, exist_ok=True)
        os.makedirs(predict_save_dir, exist_ok=True)
        os.makedirs(eval_save_dir, exist_ok=True)

        pretrain_weights_path = os.path.join(args.weights_dir, args.pretrain_weights_file)
        
        # --- Paths specific to the current dataset ---
        dataset_dir = os.path.join(args.base_data_dir, dataset_name)
            
        current_test_list = os.path.join(dataset_dir, 'test_list.txt')
        current_gt_dir = os.path.join(dataset_dir, 'gt')

        if not os.path.isdir(dataset_dir):
            logger.error(f"Dataset directory not found: {dataset_dir}. Skipping.")
            continue

        # --- 1. TRAINING ---
        logger.info(f"--- Stage 1: Training on {dataset_name} ---")
        dataset_type = get_dataset_type(dataset_name)
        logger.info(f"Using dataset type: {dataset_type} for directory: {dataset_name}")
        
        train_cmd = [
            'python', 'train.py',
            '--dataset', dataset_type,
            '--lowlight_images_path', dataset_dir,
            '--model_pretrain', pretrain_weights_path,
            '--save', train_base_dir,
            '--epochs', str(args.epochs),
            '--num_workers', str(args.num_workers)
        ]
        if not args.disable_self_ensemble:
            train_cmd.extend(['--use_self_ensemble', 'True'])
        else:
            train_cmd.extend(['--use_self_ensemble', 'False'])
        
        if args.of_model_path:
            train_cmd.extend(['--of_model_name', args.of_model_name])
            train_cmd.extend(['--of_model_path', args.of_model_path])
        
        if not run_command(train_cmd, logger):
            logger.error(f"Training failed for {dataset_name}. Skipping to next dataset.")
            continue

        # Find the actual training directory created by train.py (it has a timestamp)
        train_run_dir = find_latest_run_dir(train_base_dir)
        if not train_run_dir:
            logger.error(f"Could not find training output directory in {train_base_dir}. Skipping.")
            continue
        
        model_epochs_dir = os.path.join(train_run_dir, 'model_epochs')
        final_weights_path = find_latest_weights_file(model_epochs_dir)

        if not final_weights_path:
            logger.error(f"No weights file found in {model_epochs_dir}. Skipping.")
            continue
            
        logger.info(f"Training complete. Using final weights: {final_weights_path}")

        # --- 2. EVALUATION (Evaluation script now runs its own inference) ---
        logger.info(f"--- Stage 2: Evaluating predictions for {dataset_name} ---")
        eval_cmd = [
            'python', 'evals.py',
            '--dataset', dataset_type,
            '--lowlight_images_path', dataset_dir,
            '--model_pretrain', final_weights_path,
            '--save', eval_save_dir,
            '--name', args.eval_name
        ]
        if not args.disable_self_ensemble:
            eval_cmd.extend(['--use_self_ensemble', 'True'])
        else:
            eval_cmd.extend(['--use_self_ensemble', 'False'])
        
        if args.of_model_path:
            eval_cmd.extend(['--of_model_name', args.of_model_name])
            eval_cmd.extend(['--of_model_path', args.of_model_path])
        
        if not run_command(eval_cmd, logger):
            logger.error(f"Evaluation failed for {dataset_name}. Skipping to next dataset.")
            continue
            
        logger.info(f"Evaluation complete. Reports saved in: {eval_save_dir}")

        # --- 3. LOG FINAL RESULTS ---
        summary_json_path = os.path.join(eval_save_dir, f'{args.eval_name}_Metrics.json')
        if os.path.exists(summary_json_path):
            with open(summary_json_path, 'r') as f:
                metrics = json.load(f)
            df = pd.DataFrame([metrics])
            logger.info(f"--- FINAL PERFORMANCE for {dataset_name} ---")
            logger.info("\n" + df.to_string())
        else:
            logger.warning(f"Could not find summary file to log final results: {summary_json_path}")

        logger.info(f"========== FINISHED DATASET: {dataset_name} ==========\n")

    logger.info("Pipeline has completed for all datasets.")

    end_time = time.time()
    total_time_seconds = end_time - start_time
    logger.info(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(total_time_seconds))}")

if __name__ == '__main__':
    main()
