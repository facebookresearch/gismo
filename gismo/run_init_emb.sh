#!/bin/bash

CODE_DIR=/private/home/baharef/inversecooking2.0/proposed_model
JOBSCRIPTS_DIR=/private/home/baharef/inversecooking2.0/proposed_model/jobscripts

setup="context-full"
name="MLP_CAT"

LOGS_DIR=/checkpoint/baharef/${setup}/${name}/oct-15/

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

queue=learnlab

# Declare an array of string with type
declare -a arr=("regular")
declare -a arr_p=("avg")
declare -a arr_i=("random")

for lr in 0.005 0.001 0.005 0.0001 0.0005 0.0001 0.00005 0.00001 0.000005; do
  for w_decay in 0.0001; do
    for hidden in 300 400 500 600; do
      for emb_d in 200 300 400 500; do
        for dropout in 0.25; do
        for nr in 400; do
        for nlayers in 2; do
        for ns in "${arr[@]}"; do
        for lm in 0.0; do
        for i in 1; do
        for pool in "${arr_p[@]}"; do
        for init in "${arr_i[@]}"; do
	        file_name="lr_${lr}_w_decay_${w_decay}_hidden_${hidden}_emb_d_${emb_d}_dropout_${dropout}_nr_${nr}_nlayers_${nlayers}_neg_sampling_${ns}_lambda_${lm}_pool_${pool}_i_${i}_context_emb_mode_avg_init_emb_${init}_with_titles_True_with_set_True"
          # This creates a slurm script to call training
          job_name="title_set"
          SLURM=${JOBSCRIPTS_DIR}/run.${job_name}.sh
          echo "#!/bin/bash" > ${SLURM}
          echo "#SBATCH --job-name=$job_name" >> ${SLURM}
          echo "#SBATCH --output=${LOGS_DIR}/stdout/${file_name}_${job_name}.%j" >> ${SLURM}
          echo "#SBATCH --error=${LOGS_DIR}/stderr/${file_name}_${job_name}.%j" >> ${SLURM}
          echo "#SBATCH --partition=$queue" >> ${SLURM}
          echo "#SBATCH --constraint=volta32gb" >> ${SLURM}
          # echo "#SBATCH --signal=USR1@600" >> ${SLURM}
          echo "#SBATCH --gres=gpu:1" >> ${SLURM}
          echo "#SBATCH --cpus-per-task=8" >> ${SLURM}
          echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
          echo "#SBATCH --gpus-per-node=1" >> ${SLURM}
          echo "#SBATCH --time=20:00:00" >> ${SLURM}
          echo "#SBATCH --nodes=1" >> ${SLURM}
          echo "conda activate inv_cooking2" >> ${SLURM}
          echo "cd $CODE_DIR" >> ${SLURM}
          echo srun --label python -u train.py name=${name} setup=${setup} lr=${lr} w_decay=${w_decay} hidden=${hidden} emb_d=${emb_d} dropout=${dropout} nr=${nr} nlayers=${nlayers} neg_sampling=${ns} lambda_=${lm} i=${i} pool=${pool} add_self_loop=False max_context=43 train_batch_size=500 val_test_batch_size=50 data_augmentation=False context_emb_mode=avg epochs=500 init_emb=${init} with_titles=True with_set=True>> ${SLURM}
          sbatch ${SLURM}
          # sleep 5
done
done
done
done
done
done
done
done
done
done       
done
done

