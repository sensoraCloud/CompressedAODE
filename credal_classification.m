function [ imprecise_classes , count_trunc_imprecise ] = credal_classification( null_mod ,  lliks , odes_class_probs_istances , eps )

%trunc odes with c_low < 0
k=size(lliks,2);
c_low=zeros(1,k);

for kk=1:k
    
    c_up(1,kk)=1 - ( (lliks(1,kk) + log(1-k*eps)) / (null_mod+log(eps)) ) ;
    
end


n_istances= size(odes_class_probs_istances,1);

count_trunc_imprecise=sum(c_up<0);
index_model=c_up>0;

if count_trunc_imprecise==k
    beep
    warning('All models were truncated!');
    imprecise_classes=NaN;
else
    
    %generate against indices
    nclass=size(odes_class_probs_istances,2);
    num_confr=nclass*nclass-nclass;
    confr=zeros(num_confr,2);
    count=1;
    for class1 = 1:nclass,
        for class2 = 1:nclass,
            if class1~=class2
                confr(count,:)=[class1 class2];
                count=count+1;
            end
        end
    end
    
    
    %constraints
    new_k=sum(index_model,2);
    A=[eye(new_k,new_k);eye(new_k,new_k)*-1;ones(1,new_k)];
    bb=[ones(new_k,1)*(1-k*eps);ones(new_k,1)*eps*-1;1-eps-(k-new_k)*eps];
    
    %start points for optimization
    start_points=(1-k*eps)*eye(new_k,new_k)+(ones(new_k,new_k).*eps - eye(new_k,new_k)*eps);
    %uniform
    start_points=[start_points ; ones(1,new_k)*((1-eps)/new_k)];
    
    %not dominate class
    imprecise_classes=mat2cell(repmat(1:nclass,n_istances,1),ones(1,n_istances),nclass);
    
    options=optimset('Algorithm','active-set');
    
    %for each istances
    for d=1:n_istances
        d
        for c=1:num_confr
            c
            %check if class is dominated
            if (sum(imprecise_classes{d,:}==confr(c,1))~=0)
                
                %odes_class_probs_istances (instances X classes X odes) ;
                alpha=reshape(odes_class_probs_istances(d,confr(c,1),index_model),1,new_k);
                beta=reshape(odes_class_probs_istances(d,confr(c,2),index_model),1,new_k);
                a=sum(alpha .* (log(eps) + null_mod - lliks(1,index_model) ) ,2);
                b=sum(beta .* (log(eps) + null_mod - lliks(1,index_model) ) ,2);
                
                fval_min=inf;
                %x_min=zeros(1,new_k);
                
                %for all start points
                for kk=1:new_k+1
                    
                    f = @(x)opt_func(x,alpha,beta,a,b);
                    try
                        [x,fval] = fmincon(f,start_points(kk,:),A,bb,[],[],ones(1,new_k)*eps,ones(1,new_k)*(1-k*eps),[],options);
                    catch ME
                        fval=Inf;
                    end
                    if fval<fval_min
                        %x_min=x;
                        fval_min=fval;
                    end
                    
                end
                
                %calculate dominance
                if fval_min>1
                    %c_2 is dominated
                    imprecise_classes{d,:}=imprecise_classes{d,1}(1,imprecise_classes{d,:}~=confr(c,2));
                end
                
                
            end
            
        end
        
        
    end
    
end

    function [z] = opt_func(x,alpha,beta,a,b)
        
        z=(( sum(alpha.*log(x)) ) - a) / (( sum(beta.*log(x)) ) - b) ;
        
    end



end

