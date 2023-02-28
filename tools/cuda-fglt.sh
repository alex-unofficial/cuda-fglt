# BATCH script for testing on HPC
# Copyright (C) 2023  Alexandros Athanasiadis
# 
# This file is part of cuda-fglt
#                                                                        
# cuda-fglt is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#                                                                        
# cuda-fglt is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#                                                                        
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>. 

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --job-name=cuda-fglt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load gcc/10.2.0 cuda/11.1.0 pkgconf/1.7.3

numthreads=32
for numblocks in 64 128 256 512 1024 2048 4096; do

    make clean && make NVCCFLAGS="-arch=sm_60" CFLAGS="-DNUMBLOCKS=${numblocks} -DNUMTHREADS=${numthreads} -O3";

    result_file=data/results/blocks/result_${numblocks}_${numthreads}.txt

    ./bin/cuda-fglt data/auto/auto.mtx >> $result_file;
    ./bin/cuda-fglt data/delaunay_n22/delaunay_n22.mtx >> $result_file;
    ./bin/cuda-fglt data/great-britain_osm/great-britain_osm.mtx >> $result_file;
    ./bin/cuda-fglt data/com-Youtube/com-Youtube.mtx >> $result_file;

done

numblocks=256
for numthreads in 32 64 128 256 512; do

    make clean && make NVCCFLAGS="-arch=sm_60" CFLAGS="-DNUMBLOCKS=${numblocks} -DNUMTHREADS=${numthreads} -O3";

    result_file=data/results/threads/result_${numblocks}_${numthreads}.txt

    ./bin/cuda-fglt data/auto/auto.mtx >> $result_file;
    ./bin/cuda-fglt data/delaunay_n22/delaunay_n22.mtx >> $result_file;
    ./bin/cuda-fglt data/great-britain_osm/great-britain_osm.mtx >> $result_file;
    ./bin/cuda-fglt data/com-Youtube/com-Youtube.mtx >> $result_file;

done
