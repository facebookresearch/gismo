#!/bin/bash
#SBATCH --job-name=SAGE
#SBATCH --output=/checkpoint/baharef/context-free/SAGE/oct-26//stdout/lr_0.001_w_decay_0.0001_hidden_300_emb_d_400_dropout_0.25_nr_1_nlayers_2_i_5_margin_0.1_SAGE.%j
#SBATCH --error=/checkpoint/baharef/context-free/SAGE/oct-26//stderr/lr_0.001_w_decay_0.0001_hidden_300_emb_d_400_dropout_0.25_nr_1_nlayers_2_i_5_margin_0.1_SAGE.%j
#SBATCH --partition=learnlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --nodes=1
conda activate inv_cooking
cd /private/home/baharef/inversecooking2.0/proposed_model
srun --label python -u train.py name=SAGE setup=context-free lr=0.001 w_decay=0.0001 hidden=300 emb_d=400 dropout=0.25 nr=1 nlayers=2 i=5 epochs=1000 add_self_loop=True max_context=0 margin=0.1
