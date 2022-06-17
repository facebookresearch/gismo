#!/bin/bash
#SBATCH --job-name=GIN_MLP
#SBATCH --output=/checkpoint/baharef/context-free/GIN_MLP/oct-11//stdout/lr_0.0001_w_decay_0.0001_hidden_400_emb_d_400_dropout_0.25_nr_400_nlayers_2_i_5_p_augmentation_0.4_GIN_MLP.%j
#SBATCH --error=/checkpoint/baharef/context-free/GIN_MLP/oct-11//stderr/lr_0.0001_w_decay_0.0001_hidden_400_emb_d_400_dropout_0.25_nr_400_nlayers_2_i_5_p_augmentation_0.4_GIN_MLP.%j
#SBATCH --partition=learnlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --nodes=1
conda activate inv_cooking2
cd /private/home/baharef/inversecooking2.0/proposed_model
srun --label python -u train.py name=GIN_MLP setup=context-free lr=0.0001 w_decay=0.0001 hidden=400 emb_d=400 dropout=0.25 nr=400 nlayers=2 i=5 add_self_loop=False max_context=0 data_augmentation=True p_augmentation=0.4
