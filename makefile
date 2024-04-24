models: models.c
	clang models.c -o models -I/opt/homebrew/opt/openblas/include -L/opt/homebrew/opt/openblas/lib -lopenblas  

para: para.c
	module purge
	module load slurm
	module load cpu/0.17.3b  gcc/10.2.0/npcyll4
	module load openmpi/4.1.1
	mpicc -w -o para para.c
	sbatch para.sb

clean:
	rm -f models
	rm -f para
	rm -f my_comm
	rm -f *.out
	rm -f core.*