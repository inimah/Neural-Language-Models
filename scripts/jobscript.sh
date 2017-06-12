#!/bin/bash
#SBATCH --job-name="lm_tita"
#SBATCH -N 10          
#SBATCH -p gpu        
#SBATCH -t 300:00:00  
#SBATCH -o output.out
#SBATCH -e error.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=i.nimah@tue.nl


module load cuda/8.0.44
module load cudnn/8.0-v5.1
module load gcc/5.2.0

echo "Start = `date`"
python train_bi_europarl.py
echo "Finish = `date`"

