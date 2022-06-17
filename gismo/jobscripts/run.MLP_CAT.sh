#!/bin/bash
#SBATCH --job-name=MLP_CAT
#SBATCH --output=/checkpoint/baharef/context-full/MLP_CAT/sept-30//stdout/lr_0.00005_w_decay_0.0001_hidden_400_emb_d_200_dropout_0.25_nr_400_nlayers_2_neg_sampling_regular_lambda_0.0_i_5_MLP_CAT.%j
#SBATCH --error=/checkpoint/baharef/context-full/MLP_CAT/sept-30//stderr/lr_0.00005_w_decay_0.0001_hidden_400_emb_d_200_dropout_0.25_nr_400_nlayers_2_neg_sampling_regular_lambda_0.0_i_5_MLP_CAT.%j
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --nodes=1
conda activate inv_cooking
cd /private/home/baharef/inversecooking2.0/proposed_model
srun --label python -u train.py name=MLP_CAT setup=context-full lr=0.00005 w_decay=0.0001 hidden=400 emb_d=200 dropout=0.25 nr=400 nlayers=2 neg_sampling=regular lambda_=0.0 i=5 add_self_loop=False max_context=43 train_batch_size=500 val_test_batch_size=25 with_titles=False data_augmentation=False
