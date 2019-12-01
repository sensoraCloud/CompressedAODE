function [ classes  , probs_real_class , odes_class_probs_istances ] = classification_ode( odes ,  models_param, istances, indices , lliks , dliks  )
%classification_ode( odes, models_param, istances, indeces )
%INPUT   odes               [{} .. {}] k odes
%        models_param       [ numberOdes X typeModels ] array of k
%                           coefficients for each type of model (AODE,LODE,LLODE,..)
%        istances           istances to classify
%        lliks              generative log-lik odes 
%        dliks              discriminative log-lik odes 

%        prob_classes       [ istances X class X odes] [ p(c1|m1),..,p(cC|m1) ; .. ; p(c1|mk),..,p(cC|mk) ]      
%        indices            [{[]} .. {[]} ] k index for data of varible definition
%OUTPUT  classes            [istances X models] classes of istances
%        prob_real_class    [istances X models] probability assigned to the true classes


K=size(odes,2);

M=size(models_param,2); 

data_size=size(istances,2);

classes=zeros(data_size,M+2);

num_class=odes{1}.node_sizes(1,1);

odes_class_probs=zeros(num_class,K);

probs_real_class=zeros(data_size,M+2);

for k=1:K
    odes_engine{k}= jtree_inf_engine(odes{k});
end

odes_class_probs_istances=zeros(data_size,num_class,K);

[ exp_weights_g, high_lliks_idx_g ] = bma_weights( lliks );
[ exp_weights_d, high_lliks_idx_d ] = bma_weights( dliks );
exp_g=zeros(1,K);
exp_d=zeros(1,K);
exp_g(1,high_lliks_idx_g)=exp_weights_g(1,high_lliks_idx_g);
exp_d(1,high_lliks_idx_d)=exp_weights_d(1,high_lliks_idx_d);
models_param=[models_param exp_g'  exp_d'];


for d=1:data_size
        
    for k=1:K
                       
       class_probs = get_class_prob(odes_engine{k}, istances(indices{k},d));                   
     
       odes_class_probs(:,k)=class_probs;
       
       odes_class_probs_istances(d,:,k)=class_probs;
       
    end
    
    
    
    %calculate class probabilities of each models: AODE,LLODE,LDODE,..
    class_prob_models=odes_class_probs*models_param;
    
    %add two models
    %class_prob_models=[class_prob_models zeros(num_class,2)];
    
    
    %GBMA-AODE generative
        
%     A_prod=log(odes_class_probs) .* repmat(lliks,num_class,1);
%     
%     [mins Best_indeces] = min(A_prod,[],2);
%     
%     A_sum = log(odes_class_probs) + repmat(lliks,num_class,1);
%     
%     for c=1:num_class
%         
%         ind=1:K;
%         ind=ind(ind~=Best_indeces(c,1));
%        
%         for j=ind
%            
%             class_prob_models(c,M+1)= class_prob_models(c,M+1) + exp(A_sum(c,j) - A_sum(c,Best_indeces(c,1)));
%                         
%         end 
%         
%         class_prob_models(c,M+1)=log(class_prob_models(c,M+1)+1)+ A_sum(c,Best_indeces(c,1));
%         
%     end
%     
    
    %DBMA-AODE discriminative
    
  
    
%     A_prod=log(odes_class_probs) .* repmat(dliks,num_class,1);
%     
%     [mins Best_indeces] = min(A_prod,[],2);
%     
%     A_sum = log(odes_class_probs) + repmat(dliks,num_class,1);
%     
%     for c=1:num_class
%         
%         ind=1:K;
%         ind=ind(ind~=Best_indeces(c,1));
%        
%         for j=ind
%            
%             class_prob_models(c,M+2)= class_prob_models(c,M+2) + exp(A_sum(c,j) - A_sum(c,Best_indeces(c,1)));
%                         
%         end 
%         
%         class_prob_models(c,M+2)=log(class_prob_models(c,M+2)+1)+ A_sum(c,Best_indeces(c,1));
%         
%     end   
        
    
    for m=1:M+2
        
%         if (sum(class_prob_models(:,m))==0)
%             classes(d,m)=1;
%             probs_real_class(d,m)=0;
%         else
            
            %sometimes there are more than one classe have the same
            %probability
            classes_tmp=find(class_prob_models(:,m)==max(class_prob_models(:,m)));     
            classes(d,m)=classes_tmp(1,1); 
            probs_real_class(d,m)=class_prob_models(istances{1,d},m);            
                
           
            
        %end
    end
    
       
end

%let us not allow the exp_weight range beyong 10e3 and 10e-1
% function scaled_weights=scale_weights (exp_weights)
%     exp_weights=exp_weights*1000/max(exp_weights);
%     exp_weights(exp_weights<0.1)=0.1;
%     tmp=sum(exp_weights);
%     scaled_weights=exp_weights./tmp;
% end


end



