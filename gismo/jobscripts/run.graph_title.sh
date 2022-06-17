#!/bin/bash
#SBATCH --job-name=graph_title
#SBATCH --output=/checkpoint/baharef/context-full/GIN_MLP/oct-13//stdout/lr_0.00005_w_decay_0.0001_hidden_500_emb_d_768_dropout_0.25_nr_400_nlayers_2_neg_sampling_regular_lambda_0.0_pool_avg_i_1_context_emb_mode_avg_init_emb_food_bert_with_titles_False_with_set_True_graph_title.%j
#SBATCH --error=/checkpoint/baharef/context-full/GIN_MLP/oct-13//stderr/lr_0.00005_w_decay_0.0001_hidden_500_emb_d_768_dropout_0.25_nr_400_nlayers_2_neg_sampling_regular_lambda_0.0_pool_avg_i_1_context_emb_mode_avg_init_emb_food_bert_with_titles_False_with_set_True_graph_title.%j
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
srun --label python -u train.py name=GIN_MLP setup=context-full lr=0.00005 w_decay=0.0001 hidden=500 emb_d=768 dropout=0.25 nr=400 nlayers=2 neg_sampling=regular lambda_=0.0 i=1 pool=avg add_self_loop=False max_context=43 train_batch_size=500 val_test_batch_size=50 data_augmentation=False context_emb_mode=avg epochs=500 init_emb=food_bert with_titles=False with_set=True
