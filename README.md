# DBPE
DBPE: Distributed Byte Pair Encoding on MPI to accelerate tokenization

## How to Install

On Grace, either run reqmodules.sh or load:

1. module purge
2. module load  GCC/11.3.0  OpenMPI/4.1.4
3. module load  GROMACS/2023.1
4. module load  PyTorch
5. module load  Transformers/4.29.2
6. export HF_HOME=/scratch/user/ahazra2/hf_cache
7. export TRANSFORMERS_CACHE=/scratch/user/ahazra2/hf_cache
