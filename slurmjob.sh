. ~/initMamba.sh
conda activate ptlflow
cd ./Zero-TIG/
conda env update -f environment.yml
nvidia-smi
python run_pipeline.py --datasets RLV DID_1080 3_SDSD --base_data_dir /user/work/gf19473/datasets/ --weights_dir ./weights/ --pretrain_weights_file BVI-RLV.pt --base_exp_dir ./PIPELINE_EXP --num_workers 0 --epochs 5
