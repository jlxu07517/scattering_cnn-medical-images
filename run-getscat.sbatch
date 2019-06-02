#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=4GB
#SBATCH --job-name= imageclassify
#SBATCH --mail-type=END
#SBATCH --mail-user=jx1047@nyu.edu
#SBATCH --output=scat.out

cd /scratch/jx1047/

module purge
module load anaconda3/4.3.1
source activate image
python get.py
