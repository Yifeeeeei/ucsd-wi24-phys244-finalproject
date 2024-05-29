make clean
module purge
module load slurm
# module load cpu/0.17.3b  gcc/10.2.0/npcyll4
# module load openmpi/4.1.1
# mpicc -w -o para para.c models.c -lm -lopenblas -fopenmp

module load cpu/0.15.4  gcc/9.2.0
module load openblas/0.3.10-openmp
module load openmpi/3.1.6
mpicc -w -o para para.c models.c -lm -lopenblas -fopenmp
sbatch para.sb
