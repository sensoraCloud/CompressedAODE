function [ odes, llik, indices ] = generate_odes( training, arities )
%generate_odes( training, arietes )
%   Generete odes from 
%INPUT training     dataset   data structure data must be (attributes X istances) and
%                   first row must be class specification (class ; a1 ; a2 ; .. ; a_k)
%       arities     arieties of each variable
% OUTPUT odes       [{} .. {}] k odes learnt from data
%        llik       [ 1 X K ]array of k loglikelihoods of each ode
%       indices     [{[]} .. {[]} ] k index for data of varible definition 

vars=size(training,1);

odes=cell(1,(vars-1));
llik=zeros(1,(vars-1));

dag=zeros(vars,vars);
dag(1,2:end)=ones(1,vars-1);
dag(2,3:end)=ones(1,vars-2);
indices=cell(1,vars-1);

index=1:vars;

for o=2:vars
       
    %dataset index
   
    if (o>2)
        
        tmp=index(1,2:end);
        index(1,2:end)=circshift(tmp',vars-2)';
        
    end
    
%     index(1,2)=index(1,o);
%     index(1,o)=tmp;

    indices{1,o-1}=index;    
       
    node_sizes=arities(1,index);
    
    %make the Bayes net
    ode = mk_bnet(dag, node_sizes);
    
    %istance table with variables' names
    for k=1:vars
        ode.CPD{k} = tabular_CPD(ode,k,'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu', 'dirichlet_weight', 1);
        %ode.CPD{k} = tabular_CPD(ode,k,'prior_type', 'dirichlet', 'dirichlet_type', 'unif', 'dirichlet_weight', 1);
    end
    
    %use the junction tree engine, which is the mother of all exact
    %inference algorithms
        
    %ode_engine = jtree_inf_engine(ode);
    
    odes{1,o-1}=learn_params(ode, training(index,:));
    
    %learn parameters from data
    %[odes{1,o-1} liks] = learn_params_em(ode_engine, training(index,:));
        
    score = score_dags(training(index,:), node_sizes, {dag});
    
    %llik(1,o-1)=liks(1,end);
    
    llik(1,o-1)=score;
    
end


end


