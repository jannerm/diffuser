#!/bin/bash

#SBATCH --job-name=diffusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --account=co_rail
#SBATCH --qos=rail_gpu3_normal
#SBATCH --partition=savio3_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:TITAN:1
#SBATCH -x n0258.savio3
#SBATCH -x n0145.savio3,n0260.savio3,n0261.savio3,n0175.savio3,n0258.savio3


export ARGS="$@"
echo "Args: ${ARGS}"

singularity exec -B /var/lib/dcv-gl --nv --writable-tmpfs diffuser.sif \
        /bin/bash -c "pip install -e .; nvidia-smi; ${ARGS}"
