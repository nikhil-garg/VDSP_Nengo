#!/bin/bash
#SBATCH --account=def-drod1901
#SBATCH --time=2-23:58:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
OUTDIR=~/project/out/$SLURM_JOB_ID
mkdir -p $OUTDIR
cd $SLURM_TMPDIR


module load python/3.7

virtualenv --no-download $SLURM_TMPDIR/env  # SLURM_TMPDIR is on the compute node

source $SLURM_TMPDIR/env/bin/activate


pip install matplotlib


git clone https://github.com/nengo/nengo
cd nengo
pip install -e .

pip install Nni
pip install numpy
pip install pandas
pip install Mako

cd ..

git clone https://github.com/nikhil-garg/VDSP_ocl.git
cd VDSP_ocl

python mnist_multiple_exploration_baseline_v1.py --log_file_path $OUTDIR

tar xf $SLURM_TMPDIR/VDSP_ocl -C $OUTDIR
