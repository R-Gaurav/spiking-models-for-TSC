#!/bin/bash
#SBATCH --account=ece5984
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=144:00:00
OMP_NUM_THREADS=4 python train_eval_lsnn_and_lsnn_nhdn.py --dataset=ECG5000 --epochs=5 --is_all_combs=1 --meta_cfg_dir=normalize_no_out_v_decay_yes_order_10_theta_0_10
