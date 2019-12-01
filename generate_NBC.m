function [ nbc  ] = generate_NBC(training,arities)

        vars=size(training,1);
        dag = zeros(vars,vars);
        dag(1,2:end)=1;
        node_sizes=arities;
        nbc = mk_bnet(dag, node_sizes);
        for k=1:vars
            nbc.CPD{k} = tabular_CPD(nbc,k, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu', 'dirichlet_weight', 1);
        end
        
        %nbc_engine = jtree_inf_engine(nbc);
                
        nbc = learn_params(nbc, training);
        
end

