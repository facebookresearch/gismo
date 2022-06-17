#!/bin/bash

CODE_DIR=/private/home/baharef/inversecooking2.0/proposed_model
JOBSCRIPTS_DIR=/private/home/baharef/inversecooking2.0/proposed_model/jobscripts

setup="context-full"
name="GIN_MLP"

LOGS_DIR=/checkpoint/baharef/${setup}/${name}/oct-26/

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

queue=learnlab

# Declare an array of string with type
declare -a arr_i=("random")


# for lr in 0.000005 0.00001 0.00005 0.0001 0.0005; do
for lr in 0.00005; do
  for w_decay in 0.0001; do
    for hidden in 400; do
      for emb_d in 400; do
        for dropout in 0.25; do
        for nr in 400; do
        for nlayers in 2; do
        for lm in 0.0; do
        for i in 2 3 4 5; do
        for init in "${arr_i[@]}"; do
        for p_augmentation in 0.1; do
	        file_name="lr_${lr}_w_decay_${w_decay}_hidden_${hidden}_emb_d_${emb_d}_dropout_${dropout}_nr_${nr}_nlayers_${nlayers}_lambda_${lm}_i_${i}_init_emb_${init}_with_titles_False_with_set_True_filter_False_data_augmentation_True_p_augmentation_${p_augmentation}"
          # This creates a slurm script to call training
          job_name="bidir"
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
          echo srun --label python -u train.py name=${name} setup=${setup} lr=${lr} w_decay=${w_decay} hidden=${hidden} emb_d=${emb_d} dropout=${dropout} nr=${nr} nlayers=${nlayers} lambda_=${lm} i=${i} add_self_loop=False data_augmentation=True p_augmentation=${p_augmentation} epochs=1000 init_emb=${init} with_titles=False with_set=True filter=False >> ${SLURM}
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

