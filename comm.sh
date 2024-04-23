make clean
module purge
module load slurm
module load cpu/0.17.3b  gcc/10.2.0/npcyll4
module load openmpi/4.1.1
mpicc -w -o my_comm comm.c models.c -lm
sbatch comm.sb