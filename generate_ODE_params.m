function [ models_param count_trunc_generative count_trunc_discriminative dliks] = generate_ODE_params( odes, lliks, training_data, indices, class_entr, null_lik, arities, type_discrimLik,eps )
%Generete coeficent of mixture for ode. Type: AODE, lLODE, DLODE
%INPUT  odes        [{} .. {}] k odes
%       llik        [ 1 X K ]array of k loglikelihoods of each ode
%       training_data    datset from which odes were learned
%       indices     [{[]} .. {[]} ] k index for data of varible definition
%       class_entr  entropy class dataset
%       arities     arieties of each variable
%       type_discrimLik 0: calculate discriminative likelihood for each
%                       training istances on models learned from all data
%                       1: calculate discriminative likelihood for each
%                       training istances on models learned with cross
%                       validation paradigm
% OUTPUT params     matrix ( k X 3 ) each column defines the parameters of:
%                   AODE, lLODE, DLODE

k=size(odes,2);

data_size=size(training_data,2);

uniform=ones(k,1)/k;

%normalization

number_classes=odes{1}.node_sizes(1,1); 

[a,b]=sort(lliks);
disp(['log-likelihoods: ',num2str(b)]);


for m=1:k
    %lliks(1,m)= 1 - (lliks(1,m) / null_lik) ;
    lliks(1,m)= 1 - ((lliks(1,m) + log((1-eps)/k)) /  (null_lik + log(eps) ) ) ;
end


%set null negative weights
count_trunc_generative=sum(lliks<0);
lliks(1,lliks<0)=0;


%if trunc all weight use uniform
if (count_trunc_generative==k)
    likelihoods=uniform;
else
    %normalization
    likelihoods(:,1)=lliks/sum(lliks);
end


[a,b2]=sort(likelihoods);
disp(['likelihoods compresse: ',num2str(b2')]);

%assert(sum(b~=b2')==0, 'Order violated! generative likelihoods weights');

prob_classes=zeros(data_size,number_classes,k);
probs_real_class=zeros(data_size,k);

%if exist('models_param.m','file') ~= 2


if type_discrimLik==0
        
    for m=1:k
        odes_engine{m}= jtree_inf_engine(odes{m});
    end

    
    %for each ode calculate discriminative likelihoods
    for m=1:k
        
        
        for d=1:data_size
            
            %calculate real class probabilities
            prob_classes(d,:,m) = get_class_prob(odes_engine{m}, training_data(indices{m},d));
            probs_real_class(d,m)= prob_classes(d,training_data{1,d},m);
            
            if (probs_real_class(d,m)<eps)
                probs_real_class(d,m)=eps;
            end
            
        end
        
        %discriminativeLik(m,1)= sum(log(probs_real_class(d,m)));
        
    end
    
    
else    
        
    count_data=zeros(1,k);    
    n_fold=2;
    true_classes=training_data(1,:);
    true_classes=cell2mat(true_classes');
    c = cvpartition(true_classes,'k',n_fold);
    
    for i = 1:c.NumTestSets
        
        testIdx = test(c,i);
        trainIdx = training(c,i);
                
        [ odes_cv, lliks_cv, indices_cv ] = generate_odes( training_data(:,trainIdx), arities );
        
        for m=1:k
            odes_engine{m}= jtree_inf_engine(odes_cv{m});
        end
                       
        test_data=training_data(:,testIdx);
        
        size_test=size(test_data,2);
        
        %for each ode calculate discriminative likelihoods
        for m=1:k
                       
            for d=1:size_test
                                
                count_data(1,m)=count_data(1,m)+1;
                %calculate real class probabilities
                prob_classes(count_data(1,m),:,m) = get_class_prob(odes_engine{m}, test_data(indices_cv{m},d));
                probs_real_class(count_data(1,m),m)=prob_classes(count_data(1,m),test_data{1,d},m);
                
                if (probs_real_class(count_data(1,m),m)<eps)
                    probs_real_class(count_data(1,m),m)=eps;
                end
                
            end
            
            %discriminativeLik(m,1)= sum(log(probs_real_class(d,m)));
            
        end
        
        
        
    end
    
    
end
 
discriminativeLik(:,1)= sum(log(probs_real_class(:,:)));

dliks=discriminativeLik';

[a,b]=sort(sum(probs_real_class(:,:)));
disp(['discrim sum: ',num2str(b)]);
[a,b]=sort(sum(log(probs_real_class(:,:))));
disp(['discrim sum log: ',num2str(b)]);

%discriminativeLik = discriminativeLik ./ data_size;


low=zeros(k,1);
up=zeros(k,1);


for m=1:k
    %discriminativeLik(m,1)= 1 - ( discriminativeLik(m,1) / (class_entr*-1) ) ;
    discriminativeLik(m,1)= 1 - ( (discriminativeLik(m,1) + log((1-eps)/k)) / (((class_entr*-1)*data_size)+log(eps)) ) ;
    
    low(m,1)= 1 - ( (dliks(1,m) + log(eps)) / (((class_entr*-1)*data_size)+log(eps)) ) ;
    
    up(m,1)=1 - ( (dliks(1,m) + log(1-k*eps)) / (((class_entr*-1)*data_size)+log(eps)) ) ;

    if ((discriminativeLik(m,1)<low(m,1))||(discriminativeLik(m,1)>up(m,1)))
        beep
        warning('c precise is not inside imprecise values!!');       
    end
    
end

%normalize

up(:,1)=up(:,1)/sum(up(:,1))
low(:,1)=low(:,1)/sum(low(:,1))

%set null negative weights
count_trunc_discriminative=sum(discriminativeLik<0);
discriminativeLik(discriminativeLik<0,1)=0;

%if trunc all weight use uniform
if (count_trunc_discriminative==k)
    discriminativeLik=uniform;
else
    %normalization
    discriminativeLik(:,1)=discriminativeLik(:,1)/sum(discriminativeLik(:,1));    
end

[a,b2]=sort(discriminativeLik);
disp(['discrim lik: ',num2str(b2')]);

%assert(sum(b~=b2')==0, 'Order violated! discriminative likelihoods weights');

models_param=[uniform likelihoods discriminativeLik];

%     save('models_param.mat','models_param');
%     save('prob_classes.mat','prob_classes');
%     save('probs_real_class.mat','probs_real_class');

% else
%
%     models_param=importdata('models_param.mat');
%     prob_classes=importdata('prob_classes.mat');
%     probs_real_class=importdata('probs_real_class.mat');
%
% end

end

