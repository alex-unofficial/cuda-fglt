#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=0:10:00
#SBATCH --job-name=cuda-fglt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load gcc/10.2.0 cuda/11.1.0 pkgconf/1.7.3

make NVCCFLAGS="-arch=sm_60" -s

./bin/cuda-fglt data/auto/auto.mtx
./bin/cuda-fglt data/delaunay_n22/delaunay_n22.mtx
./bin/cuda-fglt data/great-britain_osm/great-britain_osm.mtx
./bin/cuda-fglt data/com-Youtube/com-Youtube.mtx
