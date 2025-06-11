. ~/initMamba.sh
conda activate py39
cd /user/home/ra2224/Zero-TIG
nvidia-smi
python run_pipeline.py --datasets RLV DID SDSD-indoor SDSD-outdoor --base_data_dir /user/work/gf19473/ --test_list_file ./lowlight_dataset/test_list.txt --weights_dir ./weights/ --pretrain_weights_file BVI-RLV.pt --base_exp_dir ./PIPELINE_EXP --num_workers 12 --epochs 5
