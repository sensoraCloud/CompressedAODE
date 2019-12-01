function batch_aode()
%function batch_aode1()
%split has to be comprised between 1 and  10
RandStream.setDefaultStream(RandStream('mt19937ar','seed',cputime));
addpath(genpath('/homeb/corani/FullBNT-1.0.7'));
addpath(genpath('/homeb/corani/functions'));
cd /homeb/corani/dataset;
cv_runs=10;
n_fold=5;
type_discrimLik=0;

data=dlmread('marker.csv',',', 1, 0);
if size(data,1)>1000
 cv_runs=5;
 n_fold=2;
end

ODE_classification('marker.csv',n_fold,cv_runs,type_discrimLik);    
exit;
