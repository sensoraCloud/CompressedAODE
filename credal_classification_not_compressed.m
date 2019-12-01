function [ imprecise_classes ] = credal_classification_not_compressed(  lliks , odes_class_probs_istances , eps  )

n_istances= size(odes_class_probs_istances,1);

[ exp_weights, high_lliks_idx ] = bma_weights( lliks );

k=length(high_lliks_idx);
odes_class_probs_istances=odes_class_probs_istances(:,:,high_lliks_idx);
exp_weights=exp_weights(high_lliks_idx)./sum(exp_weights(high_lliks_idx));


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


%GC:we only work with the feasible models.
A_fix=[eye(k,k)*-1;ones(1,k);ones(1,k)*-1;zeros(1,k)];
A_fix=[A_fix [ones(k,1)*eps;-1;1;-1]];
b=[zeros((k)+3,1);1;-1];


%not dominate class
imprecise_classes=mat2cell(repmat(1:nclass,n_istances,1),ones(1,n_istances),nclass);


%for each istances
for d=1:n_istances
    if k==1
        [max_prob,max_idx]=max(odes_class_probs_istances(d,:));
        imprecise_classes{d,:}=max_idx;
        continue;
    end

    %d
    for c=1:num_confr
       % c
        %check if class is dominated
        if (sum(imprecise_classes{d,:}==confr(c,1))~=0)
            
            reshape(odes_class_probs_istances(d,confr(c,1),:),1,k);
            gamma=reshape(odes_class_probs_istances(d,confr(c,1),:),1,k) .* exp_weights;
            delta=reshape(odes_class_probs_istances(d,confr(c,2),:),1,k) .* exp_weights;
           
            f=[gamma';0];
          
            A=[A_fix ; [delta 0]; [delta*-1 0]];
            
                      
            [x,fval,exitflag,output,lambda] = linprog(f,A,b);               
       
            %check
            
%             options=optimset('Algorithm','active-set');
%              
%             f_frat = @(x)opt_func(x,gamma,delta);
%             
%             A_frat=[eye(k,k);eye(k,k)*-1;ones(1,k);ones(1,k)*-1];
%             b_frat=[ones(k,1);zeros(k,1);1;-1];
%             
%             [x,fval_frat] = fmincon(f_frat,ones(1,k)/k,A_frat,b_frat,[],[],[],[],[],[]);
%             
%             if (fval_frat<fval)
%                 beep
%                 warning('fval_frat<fval!!!');
%             end
            
            %end chek
            
            
            %calculate dominance
            %note that for numerical reason we avoid a sharp 1, which is
            %instead largely augmented
            if fval>1.00001
                %c_2 is dominated
                imprecise_classes{d,:}=imprecise_classes{d,1}(1,imprecise_classes{d,:}~=confr(c,2));
            end
            
            
        end
        
    end
    
    
end

end
