#!/bin/bash

if [ $USER == adrianars ]
then
  CODE_DIR=/private/home/adrianars/code/inversecooking2.0/
  JOBSCRIPTS_DIR=/private/home/adrianars/jobscripts/inversecooking2.0/
  LOGS_DIR=/private/home/adrianars/logs/inversecooking2.0/
fi

mkdir -p ${JOBSCRIPTS_DIR}
mkdir -p ${LOGS_DIR}/stdout
mkdir -p ${LOGS_DIR}/stderr

queue=learnfair
job_name=recipe1m_im2ingr_of1

# This creates a slurm script to call training
SLURM=${JOBSCRIPTS_DIR}/run.${job_name}.slrm
echo "#!/bin/bash" > ${SLURM}
echo "#SBATCH --job-name=$job_name" >> ${SLURM}
echo "#SBATCH --output=${LOGS_DIR}/stdout/${job_name}.%j" >> ${SLURM}
echo "#SBATCH --error=${LOGS_DIR}/stderr/${job_name}.%j" >> ${SLURM}
echo "#SBATCH --partition=$queue" >> ${SLURM}
echo "#SBATCH --signal=SIGUSR1@90" >> ${SLURM}
echo "#SBATCH --gres=gpu:volta:2" >> ${SLURM}
echo "#SBATCH --cpus-per-task=8" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=64000" >> ${SLURM}
echo "#SBATCH --time=2-00:00:00" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}

echo "cd $CODE_DIR/src" >> ${SLURM}
echo srun --label python main.py >> ${SLURM}

sbatch ${SLURM}