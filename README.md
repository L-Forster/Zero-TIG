


### Finetuning optical flow models on synthetic noise.
- Contains scripts for finetuning optical flow models with synthetic noise on the Sintel dataset.
1. `finetune_of_noise/train_noisy_dpflow.py` DPFlow (Forward flow) - Simulates denoising process of I(t-1) based on the illumination channel, and I(t) using Gaussian blur, simulating the denoising kernel
2. `finetune_of_noise/train_noisy_dpflow_bwd.py` DPFlow (Backward flow) - Both I(t) and I(t+1) are processed the same way that I(t) is above.
3. `finetune_of_noise/train_noisy_raft.py RAFT` (Forward flow) - Same as (1) with RAFT model.

Example output flow map:
<img width="1676" height="325" alt="image" src="https://github.com/user-attachments/assets/c908beb1-b262-40f3-8ccf-d7f7c5d68b70" />
- Noisy version more accurately can identify the foreground in a lowlight environment

Residual (MSE) for dpflow: 2.9417	Residual (MSE) for dpflow_finetuned: 2.7472						Residual (MSE) for raft: 2.8668

### Bidirectional Image Warping
```model.py```

#### CHANGES:
- **Forward-backward temporal consistency:** Pixels that do not appear in both forward and backward warping of a frame are added to occlusion map. 
- **Bidirectional optical flow:** Calculates optical flow for I(t-1)->I(t) and I(t+1)->I(t) using the respective finetuned DPFlow models, then uses the calculated occluded regions to blend together a best estimate for the warped image. If a region is occluded in both directions, it falls back to L2.


### Optical flow & warping visualisation for all models
`demo.py`



### Running the scripts
```run_pipeline.py```

#### Examples:

**Full training + evaluation:**
python run_pipeline.py --weights_dir ./weights/ --pretrain_weights_file BVI-RLV.pt --base_exp_dir ./results/ --num_workers 12 --epochs 5 --of_model_name dpflow --of_model_path ./weights/dpflow-sintel-enhancement-finetuned.pth  --data_root ./lowlight_dataset/  --of_model_path_bwd ./weights/dpflow-sintel-noisy-backward-finetuned.pth --of_model_name_bwd dpflow

**Evaluation only:**
--evaluation_only --pretrain_weight_file [weights.pt]

**Training on specific sequence:**
--target_sequence [path_to_seq]


Results using these models on BVI_RLV dataset:

<img width="849" height="163" alt="image" src="https://github.com/user-attachments/assets/cc8b1f4e-c4f2-4d8b-a605-97802cc28618" />



### Suggested Improvements:

#### Hyperparemeter optimisation:
Occlusion / Warping hyperparameters:
 - occlusion_threshold (trained: 0.5)
 - flow_consistency_alpha (trained: 0.01)
 - fusion_confidence_threshold (trained: 0.1)

Training Noisy optical flow mdoels:
- noise_probability
- noise_params_range (alpha_brightness, gamma_brightness, band_noise, ...)
- lr (trained: 5e-5)
- 

General:
 - Epochs (trained: 5)
 - Changing loss fucntion: *I found that the loss was not reflecting the model metrics (especially the PSNR)*



```
 Starting sequence pipeline with arguments: Namespace(data_root='./lowlight_dataset/', list_file='train_list.txt', dataset='RLV', weights_dir='./weights/', pretrain_weights_file='BVI-RLV.pt', base_exp_dir='./results/', evaluation_only=False, num_workers=12, epochs=5, of_model_path='./weights/dpflow-sintel-enhancement-finetuned.pth', of_model_name='dpflow', of_model_path_bwd='./weights/dpflow-sintel-noisy-backward-finetuned.pth', of_model_name_bwd='dpflow', of_scale=3, occlusion_threshold=0.5, flow_consistency_alpha=0.01, fusion_confidence_threshold=0.1, disable_bidirectional_warp=False, target_sequence=None)
```
