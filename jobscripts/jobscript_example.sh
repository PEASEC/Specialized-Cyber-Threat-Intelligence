#!/bin/sh
##Slurm parameters
#SBATCH -J example_script_v1
#SBATCH -o outputpath/%x.out.txt
#SBATCH -e outputpath/%x.err.txt
#SBATCH -n 2
#SBATCH --mem-per-cpu=5000
#SBATCH -t 00:20:00
#SBATCH --gres=gpu
##SBATCH --mail-type=END


#---------------------------------------------------------------
# Needed for the TU Darmstadt Lichtenberg cluster to use the graphic cards
module purge
module load gcc
module load python
module load cuda
module load cuDNN

eval "$(path to the conda or minidocnda bin directory/bin/conda shell.bash hook)"
conda activate # name of your env

# The parameters are shown 
python ../src/main.py  \
    -f true -a false -b  "model name (available on hugginface.hub) or path to a model" \
    -t false -m 1 -jn test -e 1 -bs 4 -nb 10 -ee 10 \
    -jn $SLURM_JOB_NAME \
    -ed "directory path to save the experiment outputs"
    

