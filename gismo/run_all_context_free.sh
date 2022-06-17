#!/bin/bash

CODE_DIR=/private/home/baharef/inversecooking2.0/proposed_model
JOBSCRIPTS_DIR=/private/home/baharef/inversecooking2.0/proposed_model/jobscripts

setup="context-free"
name="GIN"

LOGS_DIR=/checkpoint/baharef/${setup}/${name}/oct-26/

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

queue=learnlab


for lr in 0.005 0.001 0.0005 0.0001 0.00005 0.00001; do
  for w_decay in 0.0001; do
    for hidden in 300 400 500; do
      for emb_d in 300 400 500; do
        for margin in 0.05 0.1 0.5; do
        for dropout in 0.25; do
        for nr in 1; do
        for nlayers in 2; do
        for i in 1; do
	        file_name="lr_${lr}_w_decay_${w_decay}_hidden_${hidden}_emb_d_${emb_d}_dropout_${dropout}_nr_${nr}_nlayers_${nlayers}_i_${i}_margin_${margin}"
          # This creates a slurm script to call training
          job_name="${name}"
          SLURM=${JOBSCRIPTS_DIR}/run.${job_name}.sh
          echo "#!/bin/bash" > ${SLURM}
          echo "#SBATCH --job-name=$job_name" >> ${SLURM}
          echo "#SBATCH --output=${LOGS_DIR}/stdout/${file_name}_${job_name}.%j" >> ${SLURM}
          echo "#SBATCH --error=${LOGS_DIR}/stderr/${file_name}_${job_name}.%j" >> ${SLURM}
          echo "#SBATCH --partition=$queue" >> ${SLURM}
        #   echo "#SBATCH --constraint=volta32gb" >> ${SLURM}
          # echo "#SBATCH --signal=USR1@600" >> ${SLURM}
          echo "#SBATCH --gres=gpu:1" >> ${SLURM}
          echo "#SBATCH --cpus-per-task=8" >> ${SLURM}
          echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
          echo "#SBATCH --gpus-per-node=1" >> ${SLURM}
          echo "#SBATCH --time=20:00:00" >> ${SLURM}
          echo "#SBATCH --nodes=1" >> ${SLURM}
          echo "conda activate inv_cooking" >> ${SLURM}
          echo "cd $CODE_DIR" >> ${SLURM}
          echo srun --label python -u train.py name=${name} setup=${setup} lr=${lr} w_decay=${w_decay} hidden=${hidden} emb_d=${emb_d} dropout=${dropout} nr=${nr} nlayers=${nlayers} i=${i} epochs=1000 add_self_loop=True max_context=0 margin=${margin}>> ${SLURM}

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

