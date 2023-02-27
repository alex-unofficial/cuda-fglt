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
