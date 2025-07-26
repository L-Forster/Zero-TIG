import os
import sys
import glob
import logging
import argparse
import subprocess
import pandas as pd
import json
import time
import re
import numpy as np


def setup_logging(log_file):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format=log_format,
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler(sys.stdout)])
    return logging.getLogger()

def find_latest_run_dir(base_dir):
    list_of_dirs = glob.glob(os.path.join(base_dir, 'Train-*'))
    if not list_of_dirs:
        return None
    return max(list_of_dirs, key=os.path.getctime)

def find_latest_weights_file(model_epochs_dir):
    weights_files = glob.glob(os.path.join(model_epochs_dir, 'weights_*.pt'))
    if not weights_files:
        return None
    latest_file = max(weights_files, key=lambda f: int(re.search(r'weights_(\d+).pt', f).group(1)))
    return latest_file

def run_command(command, logger):
    logger.info(f"Executing command: {' '.join(command)}")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, cwd=script_dir)
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
    parser = argparse.ArgumentParser(description="Run fine-tuning and evaluation for each video sequence in a list.")
    parser.add_argument('--data_root', type=str, required=True,
                        help="The root directory containing the dataset folders (e.g., ./data/lowlight_dataset/).")
    parser.add_argument('--list_file', type=str, default='train_list.txt',
                        help="The name of the file in data_root that lists the sequences to process.")
    parser.add_argument('--dataset', type=str, default='RLV',
                        help="The name of the dataset loader to use (e.g., RLV, DID, SDSD).")
    parser.add_argument('--weights_dir', type=str, default='./weights/',
                        help="Directory containing pre-trained model weights.")
    parser.add_argument('--pretrain_weights_file', type=str, default='BVI-RLV.pt',
                        help="Filename of the weights. Used for fine-tuning or directly for evaluation if --evaluation_only is set.")
    parser.add_argument('--base_exp_dir', type=str, default='./PIPELINE_EXP_SEQUENCES',
                        help="Base directory to save all outputs from the pipeline.")
    parser.add_argument('--evaluation_only', action='store_true',
                        help="If specified, skips the training stage and uses the --pretrain_weights_file directly for evaluation.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--of_model_path', type=str, default=None)
    parser.add_argument('--of_model_name', type=str, default='dpflow')
    parser.add_argument('--of_model_path_bwd', type=str, default=None)
    parser.add_argument('--of_model_name_bwd', type=str, default='raft')
    parser.add_argument('--of_scale', type=int, default=3)
    parser.add_argument('--occlusion_threshold', type=float, default=0.5)
    parser.add_argument('--flow_consistency_alpha', type=float, default=0.01)
    parser.add_argument('--fusion_confidence_threshold', type=float, default=0.1)
    parser.add_argument('--disable_bidirectional_warp', action='store_true')
    parser.add_argument('--target_sequence', type=str, default=None,
                        help="If specified, run the pipeline on only this sequence, ignoring the list file.")

    args = parser.parse_args()

    os.makedirs(args.base_exp_dir, exist_ok=True)
    logger = setup_logging(os.path.join(args.base_exp_dir, 'pipeline_log.txt'))

    logger.info(f"Starting sequence pipeline with arguments: {args}")
    start_time = time.time()

    
    if not args.target_sequence:
        
        train_base_dir = os.path.join(args.base_exp_dir, 'training_full_dataset')
        os.makedirs(train_base_dir, exist_ok=True)
        pretrain_weights_path = os.path.join(args.weights_dir, args.pretrain_weights_file)

        train_cmd = [
            'python', 'train.py',
            '--dataset', args.dataset,
            '--lowlight_images_path', args.data_root, 
            '--model_pretrain', pretrain_weights_path,
            '--save', train_base_dir,
            '--epochs', str(args.epochs),
            '--num_workers', str(args.num_workers)
        ]
        if args.of_model_path: train_cmd.extend(['--of_model_path', args.of_model_path])
        if args.of_model_name: train_cmd.extend(['--of_model_name', args.of_model_name])
        if args.of_model_path_bwd: train_cmd.extend(['--of_model_path_bwd', args.of_model_path_bwd])
        if args.of_model_name_bwd: train_cmd.extend(['--of_model_name_bwd', args.of_model_name_bwd])

        if not run_command(train_cmd, logger):
            logger.error("Full dataset training failed. Exiting.")
            sys.exit(1)

        train_run_dir = find_latest_run_dir(train_base_dir)
        model_epochs_dir = os.path.join(train_run_dir, 'model_epochs')
        final_weights_path = find_latest_weights_file(model_epochs_dir)

        if not final_weights_path:
            logger.error("Could not find weights file after full training. Exiting.")
            sys.exit(1)

        logger.info(f"--- Evaluating the single trained model on the test set ---")
        eval_save_dir = os.path.join(args.base_exp_dir, 'evaluation_full_dataset')
        os.makedirs(eval_save_dir, exist_ok=True)
        
        eval_cmd = [
            'python', 'evals.py',
            '--dataset', args.dataset,
            '--lowlight_images_path', args.data_root,
            '--model_pretrain', final_weights_path,
            '--save', eval_save_dir,
            '--name', 'eval_single_model'
        ]
        if args.of_model_path: eval_cmd.extend(['--of_model_path', args.of_model_path])
        if args.of_model_name: eval_cmd.extend(['--of_model_name', args.of_model_name])
        if args.of_model_path_bwd: eval_cmd.extend(['--of_model_path_bwd', args.of_model_path_bwd])
        if args.of_model_name_bwd: eval_cmd.extend(['--of_model_name_bwd', args.of_model_name_bwd])

        eval_cmd.extend(['--of_scale', str(args.of_scale)])
        if args.disable_bidirectional_warp:
            eval_cmd.append('--disable_bidirectional_warp')
        eval_cmd.extend(['--occlusion_threshold', str(args.occlusion_threshold)])
        eval_cmd.extend(['--flow_consistency_alpha', str(args.flow_consistency_alpha)])
        eval_cmd.extend(['--fusion_confidence_threshold', str(args.fusion_confidence_threshold)])

        if not run_command(eval_cmd, logger):
            logger.error("Evaluation of the single model failed.")
        else:
            logger.info(f"Evaluation complete. Reports saved in {eval_save_dir}")

    else:
        
        if args.target_sequence:
            sequence_name = args.target_sequence
            input_base_path = os.path.join(args.data_root, 'input')
            normalized_sequence_path = os.path.normpath(sequence_name)
            normalized_base_path = os.path.normpath(input_base_path)
            if normalized_sequence_path.startswith(normalized_base_path):
                sequence_name = os.path.relpath(normalized_sequence_path, normalized_base_path)
            sequences = [sequence_name.replace('\\', '/')]
            logger.info(f"Running pipeline for single target sequence: {sequences[0]}")
        else:
            sequence_list_path = os.path.join(args.data_root, args.list_file)
            try:
                with open(sequence_list_path, 'r') as f:
                    sequences = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(sequences)} sequences from {sequence_list_path}")
            except FileNotFoundError:
                logger.error(f"Sequence list file not found: {sequence_list_path}")
                sys.exit(1)

        all_metrics = []
        for seq in sequences:
            
            safe_seq_name = seq.replace('/', '_').replace('\\', '_').strip('_')
            eval_save_dir = os.path.join(args.base_exp_dir, safe_seq_name, 'evaluation')
            os.makedirs(eval_save_dir, exist_ok=True)
            
            final_weights_path = None

            if args.evaluation_only:
                logger.info("--- Stage 1: SKIPPED (evaluation_only mode) ---")
                if os.path.exists(args.pretrain_weights_file):
                     final_weights_path = args.pretrain_weights_file
                else:
                     final_weights_path = os.path.join(args.weights_dir, args.pretrain_weights_file)
                
                logger.info(f"Using provided weights for evaluation: {final_weights_path}")
                
                if not os.path.exists(final_weights_path):
                    logger.error(f"Weights file not found: {final_weights_path}. Skipping sequence.")
                    continue
            else:
                train_base_dir = os.path.join(args.base_exp_dir, safe_seq_name, 'training')
                os.makedirs(train_base_dir, exist_ok=True)
                pretrain_weights_path = os.path.join(args.weights_dir, args.pretrain_weights_file)

                logger.info(f"--- Stage 1: Fine-tuning on {seq} ---")
                
                train_cmd = [
                    'python', 'train.py',
                    '--dataset', args.dataset,
                    '--lowlight_images_path', args.data_root,
                    '--target_sequence', seq,
                    '--model_pretrain', pretrain_weights_path,
                    '--save', train_base_dir,
                    '--epochs', str(args.epochs),
                    '--num_workers', str(args.num_workers)
                ]
                if args.of_model_path: train_cmd.extend(['--of_model_path', args.of_model_path])
                if args.of_model_name: train_cmd.extend(['--of_model_name', args.of_model_name])
                if args.of_model_path_bwd: train_cmd.extend(['--of_model_path_bwd', args.of_model_path_bwd])
                if args.of_model_name_bwd: train_cmd.extend(['--of_model_name_bwd', args.of_model_name_bwd])
                if args.disable_bidirectional_warp: train_cmd.append('--disable_bidirectional_warp')
                train_cmd.extend(['--occlusion_threshold', str(args.occlusion_threshold)])
                train_cmd.extend(['--flow_consistency_alpha', str(args.flow_consistency_alpha)])
                train_cmd.extend(['--fusion_confidence_threshold', str(args.fusion_confidence_threshold)])

                if not run_command(train_cmd, logger):
                    logger.error(f"Training failed for {seq}. Skipping to next sequence.")
                    continue

                train_run_dir = find_latest_run_dir(train_base_dir)
                if not train_run_dir:
                    logger.error(f"Could not find training output directory for {seq}. Skipping.")
                    continue
                
                model_epochs_dir = os.path.join(train_run_dir, 'model_epochs')
                final_weights_path = find_latest_weights_file(model_epochs_dir)
                
                logger.info(f"Training complete for {seq}. Using final weights: {final_weights_path}")

            if not final_weights_path:
                logger.error(f"Could not find weights file to use for evaluation of {seq}. Skipping.")
                continue

            logger.info(f"--- Stage 2: Evaluating predictions for {seq} ---")
            eval_cmd = [
                'python', 'evals.py',
                '--dataset', args.dataset,
                '--lowlight_images_path', args.data_root,
                '--target_sequence', seq,
                '--model_pretrain', final_weights_path,
                '--save', eval_save_dir,
                '--name', f'eval_{safe_seq_name}'
            ]
            if args.of_model_path: eval_cmd.extend(['--of_model_path', args.of_model_path])
            if args.of_model_name: eval_cmd.extend(['--of_model_name', args.of_model_name])
            if args.of_model_path_bwd: eval_cmd.extend(['--of_model_path_bwd', args.of_model_path_bwd])
            if args.of_model_name_bwd: eval_cmd.extend(['--of_model_name_bwd', args.of_model_name_bwd])
            eval_cmd.extend(['--of_scale', str(args.of_scale)])
            if args.disable_bidirectional_warp: eval_cmd.append('--disable_bidirectional_warp')
            eval_cmd.extend(['--occlusion_threshold', str(args.occlusion_threshold)])
            eval_cmd.extend(['--flow_consistency_alpha', str(args.flow_consistency_alpha)])
            eval_cmd.extend(['--fusion_confidence_threshold', str(args.fusion_confidence_threshold)])

            if not run_command(eval_cmd, logger):
                logger.error(f"Evaluation failed for {seq}. Skipping to next sequence.")
                continue
                
            logger.info(f"Evaluation complete for {seq}. Reports saved in: {eval_save_dir}")

            summary_json_path = os.path.join(eval_save_dir, f'eval_{safe_seq_name}_Metrics.json')
            if os.path.exists(summary_json_path):
                with open(summary_json_path, 'r') as f:
                    metrics = json.load(f)
                    metrics['sequence'] = seq
                    all_metrics.append(metrics)
            else:
                logger.warning(f"Could not find summary file for {seq}: {summary_json_path}")

        if all_metrics:
            df = pd.DataFrame(all_metrics)
            cols = ['sequence'] + [col for col in df.columns if col != 'sequence']
            df = df[cols]
            avg_metrics = df.select_dtypes(include=np.number).mean().to_frame().T
            
            logger.info("\n\n" + "="*50)
            logger.info("--- INDIVIDUAL SEQUENCE RESULTS (from per-sequence training) ---")
            logger.info("="*50 + "\n")
            logger.info("\n" + df.to_string())
            
            logger.info("\n\n" + "="*50)
            logger.info("--- AVERAGE METRICS ACROSS ALL SEQUENCES ---")
            logger.info("="*50 + "\n")
            logger.info("\n" + avg_metrics.to_string())

            avg_metrics_path = os.path.join(args.base_exp_dir, 'average_metrics_per_sequence.csv')
            df.to_csv(os.path.join(args.base_exp_dir, 'individual_metrics_per_sequence.csv'), index=False)
            avg_metrics.to_csv(avg_metrics_path, index=False)
            logger.info(f"\nSaved per-sequence average metrics to {avg_metrics_path}")
        else:
            logger.warning("No metrics were collected. Cannot calculate averages.")


    logger.info("Pipeline has completed.")
    end_time = time.time()
    total_time_seconds = end_time - start_time
    logger.info(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(total_time_seconds))}")


if __name__ == "__main__":
    main()