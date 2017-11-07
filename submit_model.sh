#!/bin/sh
#SBATCH --output=/work/yd44/imputation/log/model.log
#SBATCH --job-name=mtest
#SBATCH --error=/work/yd44/imputation/log/model.error
#SBATCH --mem=5G
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1

source /opt/apps/sdg/sdg_bashrc
source /dscrhome/yd44/.bashrc
source /dscrhome/yd44/software/virtualpy/tensorflowGPU/bin/activate

### vcf for chr5 found
python3 /dscrhome/yd44/imputation/script/imptest/model.py



