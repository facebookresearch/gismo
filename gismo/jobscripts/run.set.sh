#!/bin/bash
#SBATCH --job-name=set
#SBATCH --output=/checkpoint/baharef/context-full/GIN_MLP/oct-26//stdout/lr_0.00005_w_decay_0.0001_hidden_300_emb_d_300_dropout_0.25_nr_400_nlayers_2_lambda_0.0_i_5_init_emb_random_with_titles_False_with_set_True_filter_False_set.%j
#SBATCH --error=/checkpoint/baharef/context-full/GIN_MLP/oct-26//stderr/lr_0.00005_w_decay_0.0001_hidden_300_emb_d_300_dropout_0.25_nr_400_nlayers_2_lambda_0.0_i_5_init_emb_random_with_titles_False_with_set_True_filter_False_set.%j
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --nodes=1
conda activate inv_cooking2
cd /private/home/baharef/inversecooking2.0/proposed_model
srun --label python -u train.py name=GIN_MLP setup=context-full lr=0.00005 w_decay=0.0001 hidden=300 emb_d=300 dropout=0.25 nr=400 nlayers=2 lambda_=0.0 i=5 add_self_loop=False data_augmentation=False p_augmentation=0.5 epochs=1000 init_emb=random with_titles=False with_set=True filter=False
