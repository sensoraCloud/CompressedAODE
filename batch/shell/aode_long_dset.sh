#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -m eas
#$ -M giorgio@idsia.ch
#$ -l h_rt=23:59:59
#$ -l h_vmem=2G
cd /homeb/corani/functions/batch;
script_to_run=$1;
#echo $script_to_run;
/usr/local/bin/matlab -singleCompThread -nodisplay -nosplash -r $script_to_run;

