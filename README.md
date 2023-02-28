# cuda-fglt
## computes the FGLT of a graph using CUDA

This program was built as an assignment for my Parallel & Distributed Comp. Systems
class in University.

Compiling
---------
clone the repository with
```bash
git clone https://github.com/alex-unofficial/cuda-fglt
```

### local setup
you will need to have installed [CUDA](https://developer.nvidia.com/cuda-toolkit) on your system, 
as well as [gcc](https://gcc.gnu.org/) and [pkgconf](http://pkgconf.org/).

then simply run
```bash
make
```

### setp on `AUTH IT Compute Cluster`
you will need to run the following line before compiling
```
module load gcc/10.2.0 cuda/11.1.0 pkgconf/1.7.3
```

Then run
```bash
make NVCCFLAGS="-arch=sm_60"
```
if running on the `gpu` partition.


Running
-------
to run the program fo
```bash
./bin/cuda-fglt matrix-file.mtx
```

Licence
-------
```
Copyright (C) 2023  Alexandros Athanasiadis

This file is part of cuda-fglt

cuda-fglt is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

knn is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

```
