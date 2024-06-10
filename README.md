# How to run parallel version on expanse
We have a one-click script that sets up the enviroment and commit the job
To use it, run the following command:
sh para.sh

The corresponding .sb file is para.sb

# How to run serial version on expance
We also have a one-click script that sets up the enviroment and commit the job
To use it, run the following command:
sh seri.sh

The corresponding .sb file is seri.sb

**Attention: the script will clean all previous .out files in this directory, make sure you move them somewhere else if you want to keep them**


# Results
The performances are measured in microseconds, and results are placed in ./outputs

To plot the results into figures
cd ./outputs
python plot.py