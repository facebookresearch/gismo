#!/bin/bash
#SBATCH --job-name=GIN
#SBATCH --output=/checkpoint/baharef/context-free/GIN/oct-26//stdout/lr_0.00001_w_decay_0.0001_hidden_500_emb_d_500_dropout_0.25_nr_1_nlayers_2_i_1_margin_0.5_GIN.%j
#SBATCH --error=/checkpoint/baharef/context-free/GIN/oct-26//stderr/lr_0.00001_w_decay_0.0001_hidden_500_emb_d_500_dropout_0.25_nr_1_nlayers_2_i_1_margin_0.5_GIN.%j
#SBATCH --partition=learnlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --nodes=1
conda activate inv_cooking
cd /private/home/baharef/inversecooking2.0/proposed_model
srun --label python -u train.py name=GIN setup=context-free lr=0.00001 w_decay=0.0001 hidden=500 emb_d=500 dropout=0.25 nr=1 nlayers=2 i=1 epochs=1000 add_self_loop=True max_context=0 margin=0.5
