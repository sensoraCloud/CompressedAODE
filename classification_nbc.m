function [ classes, probs_real_class ] = classification_nbc( nbc, istances )
%classification_nbc( nbc, istances )
%INPUT   nbc                naive bayes classifier
%        istances           istances to classify
%OUTPUT  classes            [istances X 1] classes of istances
%        prob_real_class    [istances X 1] probability assigned to the true classes

data_size=size(istances,2);

classes=zeros(data_size,1);
probs_real_class=zeros(data_size,1);
nbc_engine=jtree_inf_engine(nbc);

for d=1:data_size
    
    class_probs = get_class_prob(nbc_engine, istances(:,d));
    
    classes_tmp=find(class_probs==max(class_probs));  
            
    classes(d,1)=classes_tmp(1,1); 
    
    probs_real_class(d,1)=class_probs(istances{1,d},1);
    
end


end

