#!/bin/bash
#SBATCH --job-name=title_set
#SBATCH --output=/checkpoint/baharef/context-full/MLP_CAT/oct-22//stdout/lr_0.001_w_decay_0.0001_hidden_600_emb_d_600_dropout_0.25_nr_400_nlayers_2_neg_sampling_regular_lambda_0.0_pool_avg_i_1_context_emb_mode_avg_init_emb_random_with_titles_True_with_set_True_title_set.%j
#SBATCH --error=/checkpoint/baharef/context-full/MLP_CAT/oct-22//stderr/lr_0.001_w_decay_0.0001_hidden_600_emb_d_600_dropout_0.25_nr_400_nlayers_2_neg_sampling_regular_lambda_0.0_pool_avg_i_1_context_emb_mode_avg_init_emb_random_with_titles_True_with_set_True_title_set.%j
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
srun --label python -u train.py name=MLP_CAT setup=context-full lr=0.001 w_decay=0.0001 hidden=600 emb_d=600 dropout=0.25 nr=400 nlayers=2 neg_sampling=regular lambda_=0.0 i=1 pool=avg add_self_loop=False max_context=43 train_batch_size=500 val_test_batch_size=50 data_augmentation=False p_augmentation=0.5 context_emb_mode=avg epochs=500 init_emb=random with_titles=True with_set=True
