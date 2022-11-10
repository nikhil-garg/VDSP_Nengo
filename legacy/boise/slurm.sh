#!/bin/bash
#SBATCH --account=def-drod1901
#SBATCH --time=0-12:0:0
#SBATCH --cpus-per-task=32 
#SBATCH --mem=25G
OUTDIR=~/project/out/$SLURM_JOB_ID
mkdir -p $OUTDIR
cd $SLURM_TMPDIR


module load python/3.7

virtualenv --no-download $SLURM_TMPDIR/env  # SLURM_TMPDIR is on the compute node

source $SLURM_TMPDIR/env/bin/activate


pip install matplotlib

git clone -b v2.0 https://github.com/Microsoft/nni.git
cd nni
python -m pip install --upgrade pip setuptools
python setup.py develop
cd ..


git clone https://github.com/nengo/nengo
cd nengo
pip install -e .

pip install nni
pip install numpy
pip install pandas
pip install Mako


cd ..

git clone https://github.com/nikhil-garg/VDSP_ocl.git
cd VDSP_ocl


nnictl create --config config.yml
nnictl experiment export --filename nni_log.csv --type csv