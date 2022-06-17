#!/bin/bash
#SBATCH --job-name=GIN_MLP2
#SBATCH --output=/checkpoint/baharef/context-full/GIN_MLP2/oct-11//stdout/lr_0.0001_w_decay_0.0001_hidden_400_emb_d_500_dropout_0.25_nr_400_nlayers_2_neg_sampling_regular_lambda_0.0_pool_min_i_1_context_emb_mode_transformer_GIN_MLP2.%j
#SBATCH --error=/checkpoint/baharef/context-full/GIN_MLP2/oct-11//stderr/lr_0.0001_w_decay_0.0001_hidden_400_emb_d_500_dropout_0.25_nr_400_nlayers_2_neg_sampling_regular_lambda_0.0_pool_min_i_1_context_emb_mode_transformer_GIN_MLP2.%j
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=50:00:00
#SBATCH --nodes=1
conda activate inv_cooking2
cd /private/home/baharef/inversecooking2.0/proposed_model
srun --label python -u train.py name=GIN_MLP2 setup=context-full lr=0.0001 w_decay=0.0001 hidden=400 emb_d=500 dropout=0.25 nr=400 nlayers=2 neg_sampling=regular lambda_=0.0 i=1 pool=min add_self_loop=False max_context=43 train_batch_size=200 val_test_batch_size=10 with_titles=False data_augmentation=False context_emb_mode=avg epochs=1000
