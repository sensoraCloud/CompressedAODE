function ODE_classification(csv_file,n_fold,cv_runs,type_discrimLik,eps,generative)
%ODE_classification(csv_file,n_fold,cv_runs,type_discrimLik,eps,generative)
%loads data from the csv file and perform classification experiments,
%generating mcar missingness with prob_missing.
%set generative to 'on' if you want to run also generative models, by
%default it is set to off

%eps=10^-2;
rand('twister',6657);

if nargin<2
    n_fold=10;
end

if nargin<3
    cv_runs=10;
end

if nargin<4
    type_discrimLik=0;
end


if nargin<5
    eps=0.01;
end

if nargin<6
    generative='off';
end

%number of models (AODE, C-GBMA-AODE, C-DBMA-AODE, GBMA-AODE , DBMA-AODE , NBC)
M=6;

init();

counter=0;

%calculate class entropy

%class entropy
% freq_class=zeros(1,arities(1,1));

data_size=size(data,1);

% for c=1:arities(1,1)
%
%     numerator=size(find(data(:,1)==c),1);
%
%     freq_class(1,c)=numerator /  data_size;
%
%     if (freq_class(1,c)==0)
%         freq_class(1,c)=1;
%     end
%
% end

%class entropy
freq_class(1,:) = accumarray(data(:,1),ones(size(data(:,1))));

freq_class(1,freq_class<eps)=eps;

freq_class=freq_class ./ data_size;

%freq_class(1,freq_class==0)=1;

class_entr = -sum(freq_class.*log(freq_class));

data=num2cell(data');

vars=size(data,1);
dag=zeros(vars,vars);
%null model likelihood
null_lik = score_dags(data, arities, {dag});

true_classes=data(1,:);
true_classes=cell2mat(true_classes');

for cv_run=1:cv_runs
    
    %Leave-one-out
    if n_fold==0
        c = cvpartition(true_classes,'leaveout');
    else
        c = cvpartition(true_classes,'k',n_fold);
    end
    
    
    for i = 1:c.NumTestSets
        
        %for each model (AODE,LLODE,..) create a class performances
        
        for m=1:M
            
            cp{m} = classperf(true_classes); % initializes the CP object
            
        end
        
        counter=counter+1;
        testIdx = test(c,i);
        trainIdx = training(c,i);
        
        nbc  = generate_NBC(data(:,trainIdx),arities);
        [ odes, lliks, indices ] = generate_odes( data(:,trainIdx), arities );
        [ models_param count_trunc_generative count_trunc_discriminative dliks] = generate_ODE_params( odes, lliks, data(:,trainIdx) , indices, class_entr, null_lik, arities, type_discrimLik,eps );
        [ classes_ODE , probs_real_class_ODE , odes_class_probs_istances ] = classification_ode( odes,  models_param,  data(:,testIdx),  indices , lliks , dliks  );
        [ classes_NBC, probs_real_class_NBC ] = classification_nbc( nbc,  data(:,testIdx) );
        
        %odes_class_probs_istances (instances X classes X models)
        
        n_istances_train=sum(trainIdx==1);
        
        %c-DCMA credal classification with discriminative
        null_model=((class_entr*-1)*n_istances_train);
        [ imprecise_classes , count_trunc_imprecise  ] = credal_classification( null_model , dliks , odes_class_probs_istances ,  eps);
        
        if count_trunc_discriminative>count_trunc_imprecise
            beep
            warning('count_trunc_discriminative>count_trunc_imprecise!!!');
        end
        
        
        if iscell(imprecise_classes)
            
            %calculate imprecise classify performances
            [ det,sing_acc,ind_out_size,set_acc,disc_acc,util_65,util_80,precise_single,precise_set, imprecise_index_c_dcma ] = imprecise_performances( imprecise_classes , true_classes(testIdx) , classes_ODE(:,3) );
            
        else
            
            %NaN performances (all models were truncated)
            det= NaN;sing_acc=NaN;ind_out_size= NaN;set_acc= NaN;disc_acc= NaN;util_65= NaN;util_80= NaN;precise_single= NaN;precise_set = NaN;
            
        end
        
        %imprecise results
        
        report_results.class_results(counter,M+M+3)=det;
        report_results.class_results(counter,M+M+4)=sing_acc;
        report_results.class_results(counter,M+M+5)=ind_out_size;
        report_results.class_results(counter,M+M+6)=set_acc;
        report_results.class_results(counter,M+M+7)=disc_acc;
        report_results.class_results(counter,M+M+8)=util_65;
        report_results.class_results(counter,M+M+9)=util_80;
        report_results.class_results(counter,M+M+10)=precise_single;
        report_results.class_results(counter,M+M+11)=precise_set;
        report_results.class_results(counter,M+M+12)=count_trunc_imprecise;
        
        
        if ((count_trunc_imprecise==0)&&(precise_single ~= sing_acc))
            beep
            warning('single_acc different precise_single!!!');
        end
        
        if strcmp(generative,'on')
            %c-GCMA credal classification with generative
            [imprecise_classes, count_trunc_imprecise] =credal_classification( null_lik , lliks , odes_class_probs_istances , eps);
            if count_trunc_discriminative>count_trunc_imprecise
                %beep
                warning('count_trunc_discriminative>count_trunc_imprecise!!!');
            end
            if iscell(imprecise_classes)
                
                %calculate imprecise classify performances
                [ det,sing_acc,ind_out_size,set_acc,disc_acc,util_65,util_80,precise_single,precise_set, imprecise_index_c_gcma ] = imprecise_performances( imprecise_classes , true_classes(testIdx) , classes_ODE(:,2) );
                
            else
                
                %NaN performances (all models were truncated)
                det= NaN;sing_acc=NaN;ind_out_size= NaN;set_acc= NaN;disc_acc= NaN;util_65= NaN;util_80= NaN;precise_single= NaN;precise_set = NaN;
                
            end
            
            %imprecise results
            
            report_results.class_results(counter,M+M+13)=det;
            report_results.class_results(counter,M+M+14)=sing_acc;
            report_results.class_results(counter,M+M+15)=ind_out_size;
            report_results.class_results(counter,M+M+16)=set_acc;
            report_results.class_results(counter,M+M+17)=disc_acc;
            report_results.class_results(counter,M+M+18)=util_65;
            report_results.class_results(counter,M+M+19)=util_80;
            report_results.class_results(counter,M+M+20)=precise_single;
            report_results.class_results(counter,M+M+21)=precise_set;
            report_results.class_results(counter,M+M+22)=count_trunc_imprecise;
            
            
            if ((count_trunc_imprecise==0)&&(precise_single ~= sing_acc))
                beep
                warning('single_acc different precise_single!!!');
            end
            
            
            %GCMA credal classification with generative (not compressed)
            [ imprecise_classes ] = credal_classification_not_compressed(  lliks , odes_class_probs_istances  , eps );
            
            %calculate imprecise classify performances
            [ det,sing_acc,ind_out_size,set_acc,disc_acc,util_65,util_80,precise_single,precise_set, imprecise_index_gcma ] = imprecise_performances( imprecise_classes , true_classes(testIdx) , classes_ODE(:,4) );
            
            
            %imprecise results
            report_results.class_results(counter,M+M+23)=det;
            report_results.class_results(counter,M+M+24)=sing_acc;
            report_results.class_results(counter,M+M+25)=ind_out_size;
            report_results.class_results(counter,M+M+26)=set_acc;
            report_results.class_results(counter,M+M+27)=disc_acc;
            report_results.class_results(counter,M+M+28)=util_65;
            report_results.class_results(counter,M+M+29)=util_80;
            report_results.class_results(counter,M+M+30)=precise_single;
            report_results.class_results(counter,M+M+31)=precise_set;
            
            %G-COMP vs BMA
            count_impr_compr=sum(imprecise_index_c_gcma);
            count_and_impr=sum(imprecise_index_c_gcma.*imprecise_index_gcma);
            if count_impr_compr==0
                report_results.class_results(counter,M+M+42)=NaN;
            else
                report_results.class_results(counter,M+M+42)=count_and_impr/count_impr_compr;
            end
            
        else
            
            report_results.class_results(counter,M+M+42)=NaN;
            
        end
        
        %DCMA credal classification with discriminative (not compressed)
        
        [ imprecise_classes ] = credal_classification_not_compressed(  dliks , odes_class_probs_istances , eps  );
        
        %calculate imprecise classify performances
        [ det,sing_acc,ind_out_size,set_acc,disc_acc,util_65,util_80,precise_single,precise_set, imprecise_index_dcma ] = imprecise_performances( imprecise_classes , true_classes(testIdx) , classes_ODE(:,5) );
        
        
        %imprecise results
        
        report_results.class_results(counter,M+M+32)=det;
        report_results.class_results(counter,M+M+33)=sing_acc;
        report_results.class_results(counter,M+M+34)=ind_out_size;
        report_results.class_results(counter,M+M+35)=set_acc;
        report_results.class_results(counter,M+M+36)=disc_acc;
        report_results.class_results(counter,M+M+37)=util_65;
        report_results.class_results(counter,M+M+38)=util_80;
        report_results.class_results(counter,M+M+39)=precise_single;
        report_results.class_results(counter,M+M+40)=precise_set;
        
        %precent imprecision
        
        %D-COMP vs BMA
        count_impr_compr=sum(imprecise_index_c_dcma);
        count_and_impr=sum(imprecise_index_c_dcma.*imprecise_index_dcma);
        
        if count_impr_compr==0
            report_results.class_results(counter,M+M+41)=NaN;
        else
            report_results.class_results(counter,M+M+41)=count_and_impr/count_impr_compr;
        end
        
        
        %for each model calculate performances
        
        for m=1:M-1
            
            classperf(cp{m},classes_ODE(:,m),testIdx);
            
        end
        
        %add NBC
        classperf(cp{M},classes_NBC,testIdx);
        
        %save accuracies
        for m=1:M-1
            
            report_results.class_results(counter,m)=cp{m}.CorrectRate;
            
        end
        
        %add NBC result
        report_results.class_results(counter,M)=cp{M}.CorrectRate;
        
        size_test=size(testIdx,1);
        
        %save brier score
        for m=1:M-1
            
            report_results.class_results(counter,M+m)=( sum( (1-probs_real_class_ODE(:,m)).*(1-probs_real_class_ODE(:,m)) ) ) / size_test;
            
        end
        
        %add NBC result
        report_results.class_results(counter,M+M)=( sum( (1-probs_real_class_NBC(:,1)).*(1-probs_real_class_NBC(:,1)) ) ) / size_test;
        
        report_results.class_results(counter,M+M+1)=count_trunc_generative;
        report_results.class_results(counter,M+M+2)=count_trunc_discriminative;
        
        save(strcat('results_',csv_file,'.mat'),'report_results');
    end
    
    save(strcat('results_',csv_file,'.mat'),'report_results');
    
end

save(strcat('results_',csv_file,'.mat'),'report_results');

report_ode_results(report_results);

%performs various inits
    function init()
        %GC
        data=dlmread(fullfile('dataset',csv_file), ',', 1, 0);
        %data=dlmread(fullfile(csv_file), ',', 1, 0);
        %let us bring the class in first position
        data=[data(:,end) data(:,1:end-1)];
        
        %we need to filter out features with a single state
        arities=max(data);
        idx=find(arities==1);
        idx2=setdiff(1:size(data,2),idx);
        data=data(:,idx2);
        arities=max(data);
        
        
        %data set is ready for experiments
        tot_cv_folds=n_fold;
        tot_trials=tot_cv_folds*cv_runs;
        
        report_results.dataset=csv_file;
        report_results.size_dataset=size(data,1);
        report_results.number_classes=arities(1);
        report_results.number_features=size(data,2)-1;
        report_results.type_disclik=type_discrimLik;
        report_results.nfold=n_fold;
        report_results.result_fields={'acc_AODE','acc_LLODE','acc_LDODE','acc_NBC','br_score_AODE','br_score_LLODE','br_score_LDODE','br_score_NBC','count_trunc_generative' , 'count_trunc_discriminative' };
        report_results.class_results=zeros(tot_trials,10);
        report_results.M=M;
        
    end
end
