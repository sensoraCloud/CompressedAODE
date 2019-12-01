function batch_aode_anneal_unsup_discr_unsup()
%function batch_aode_anneal_unsup_discr1()
%split has to be comprised between 1 and  10
RandStream.setDefaultStream(RandStream('mt19937ar','seed',0));
addpath(genpath('/homeb/corani/FullBNT-1.0.7'));
addpath(genpath('/homeb/corani/functions'));
cd /homeb/corani/dataset/unsup;
cv_runs=10;
n_fold=5;
type_discrimLik=0;

data=dlmread('anneal_unsup_discr.csv',',', 1, 0);
if size(data,1)>1000
 cv_runs=6;
end

ODE_classification('anneal_unsup_discr.csv',n_fold,cv_runs,type_discrimLik);    
exit;
