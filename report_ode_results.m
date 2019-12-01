function report_ode_results(cv)
%function report_ode_results(cv_results,em_included)
%if generated_data is not passed, it is assumed to be 0.

reportfile='report_cv.csv';
% if (generated_data)
%     reportfile='gen_data_report.csv';
% end


%GC: necessary to deal with possibly incomplete results
idx=cv.class_results(:,1)>0;
cv.class_results=cv.class_results(idx,:);


if (exist(fullfile(pwd,reportfile),'file')==0)
    
    fid = fopen(reportfile,'w');
    fprintf(fid,'%s','dataset,n,feats,classes,nfold,');
    %     if generated_data
    %         fprintf(fid,'%s %s','states',',');
    %     end
    fprintf(fid,'%s','acc_AODE, acc_C-GBMA-AODE, acc_C-DBMA-AODE, acc_GBMA-AODE, acc_DBMA-AODE, acc_NBC, br_score_AODE, br_score_C-GBMA-AODE, br_score_C-DBMA-AODE, br_score_GBMA, br_score_DBMA, br_score_NBC, p-value-acc_C-GBMA-AODE, p-value-acc_C-DBMA-AODE, p-value-acc_NBC, p-value-brier-score_GAODE, p-value-brier-score_GAODE, p-value-brier-score_NBC,count_trunc_generative, count_trunc_discriminative, Type_discr_lik, det_C-DBMA-AODE*,sing_acc_C-DBMA-AODE*,ind_out_size_C-DBMA-AODE*,set_acc_C-DBMA-AODE*,disc_acc_C-DBMA-AODE*,util_65_C-DBMA-AODE*,util_80_C-DBMA-AODE*,precise_single_C-DBMA-AODE*,precise_set_C-DBMA-AODE*,p-value-65_C-DBMA-AODE*,p-value-80_C-DBMA-AODE*,count_trunc_imprecise_C-DBMA-AODE*,det_C-GBMA-AODE*,sing_acc_C-GBMA-AODE*,ind_out_size_C-GBMA-AODE*,set_acc_C-GBMA-AODE*,disc_acc_C-GBMA-AODE*,util_65_C-GBMA-AODE*,util_80_C-GBMA-AODE*,precise_single_C-GBMA-AODE*,precise_set_C-GBMA-AODE*,p-value-65_C-GBMA-AODE*,p-value-80_C-GBMA-AODE*,count_trunc_imprecise_C-GBMA-AODE*,det_GBMA-AODE*,sing_acc_GBMA-AODE*,ind_out_size_GBMA-AODE*,set_acc_GBMA-AODE*,disc_acc_GBMA-AODE*,util_65_GBMA-AODE*,util_80_GBMA-AODE*,precise_single_GBMA-AODE*,precise_set_GBMA-AODE*,p-value-65_GBMA-AODE*,p-value-80_GBMA-AODE*, det_DBMA-AODE*,sing_acc_DBMA-AODE*,ind_out_size_DBMA-AODE*,set_acc_DBMA-AODE*,disc_acc_DBMA-AODE*,util_65_DBMA-AODE*,util_80_DBMA-AODE*,precise_single_DBMA-AODE*,precise_set_DBMA-AODE*,p-value-65_DBMA-AODE*,p-value-80_DBMA-AODE*, %D-COMPvsBMA, %G-COMPvsBMA');
    
else
    fid = fopen(reportfile,'a');
end

te_tr_ratio=1/(cv.nfold-1);
% if te_tr_ratio==1
%     te_tr_ratio=1-eps;
% end

alpha=0.05;

for i=1:length(cv)
    
    fprintf(fid,'\n%s %s %d %s %d %s %d %s %d %s',strrep(cv.dataset,'gc_',''),',',cv.size_dataset,',',cv.number_features,',',cv.number_classes,',',cv.nfold,',');
    
    %mean of accuracies and brier scores
    tmp(1:cv.M*2)=num2cell(mean(cv.class_results(:,1:cv.M*2),1));
    
    %min log-loss
    %tmp(13:16)=num2cell(mean(cv.class_results(:,9:12),1));
    
    %median of ratio of log-loss
    %     for kk=1:6
    %         tmp{10+kk}=nanmedian(cv.class_results(:,6+kk));
    %     end
    
    %resampled_ttest(x,y,te_tr_ratio,alpha)
    
    %p-value of accuracy, AODE vs LLODE
    [h,tmp{(cv.M*2)+1},ci,stats]=resampled_ttest(cv.class_results(:,1),cv.class_results(:,2),te_tr_ratio,alpha);
    %p-value of accuracy, AODE vs LDODE
    [h,tmp{(cv.M*2)+2},ci,stats]=resampled_ttest(cv.class_results(:,1),cv.class_results(:,3),te_tr_ratio,alpha);
    %p-value of accuracy, AODE vs NBC
    [h,tmp{(cv.M*2)+3},ci,stats]=resampled_ttest(cv.class_results(:,1),cv.class_results(:,4),te_tr_ratio,alpha);
    
    
    %p-value of brier scores, AODE vs LLODE
    [h,tmp{(cv.M*2)+4},ci,stats]=resampled_ttest(cv.class_results(:,5),cv.class_results(:,6),te_tr_ratio,alpha);
    %p-value of brier scores, AODE vs LDODE
    [h,tmp{(cv.M*2)+5},ci,stats]=resampled_ttest(cv.class_results(:,5),cv.class_results(:,7),te_tr_ratio,alpha);
    %p-value of brier scores, AODE vs NBC
    [h,tmp{(cv.M*2)+6},ci,stats]=resampled_ttest(cv.class_results(:,5),cv.class_results(:,8),te_tr_ratio,alpha);
    
    %mean count trunc generative
    tmp((cv.M*2)+7)=num2cell(mean(cv.class_results(:,(cv.M*2)+1),1));
    
    %mean count trunc discirminative
    tmp((cv.M*2)+8)=num2cell(mean(cv.class_results(:,(cv.M*2)+2),1));
    
    %type discirminative likelihood
    tmp((cv.M*2)+9)=num2cell(cv.type_disclik);
    
    %NaN control
    
    %     if ~isempty(find(isnan(cv.class_results(:,11:19))==1))
    %         tmp(16:27)=num2str(NaN);
    %     else
    
    
    %         if ((nanmean(cv.class_results(:,20),1)==0)&&(nanmean(cv.class_results(:,12),1))~= nanmean(cv.class_results(:,18),1))
    %             beep
    %             warning('single_acc different precise_single!!!');
    %         end
    %
    
    %C-GBMA-AODE*
    tmp(((cv.M*2)+10):((cv.M*2)+18))=num2cell(nanmean(cv.class_results(:,((cv.M*2)+3):((cv.M*2)+11)),1));
    
    if ~isempty(find(isnan(cv.class_results(:,((cv.M*2)+8):((cv.M*2)+9)))==1))
        tmp{((cv.M*2)+19):((cv.M*2)+20)}=num2str(NaN);
    else
        %p-value of accuracy, LDODE vs LLODE utility 65
        [h,tmp{((cv.M*2)+19)},ci,stats]=resampled_ttest(cv.class_results(:,((cv.M*2)+8)),cv.class_results(:,2),te_tr_ratio,alpha);
        %p-value of accuracy, LDODE vs LLODE utility 80
        [h,tmp{((cv.M*2)+20)},ci,stats]=resampled_ttest(cv.class_results(:,((cv.M*2)+9)),cv.class_results(:,2),te_tr_ratio,alpha);
    end
    
    %mean count trunc imprecise
    tmp(((cv.M*2)+21))=num2cell(nanmean(cv.class_results(:,((cv.M*2)+12)),1));
    %     end
    
    %
    
    
    %C-DBMA-AODE*
    tmp(((cv.M*2)+22):((cv.M*2)+30))=num2cell(nanmean(cv.class_results(:,((cv.M*2)+13):((cv.M*2)+21)),1));
    
    if ~isempty(find(isnan(cv.class_results(:,((cv.M*2)+18):((cv.M*2)+19)))==1))
        tmp{((cv.M*2)+31):((cv.M*2)+32)}=num2str(NaN);
    else
        %p-value of accuracy, LDODE vs LDODE utility 65
        [h,tmp{((cv.M*2)+31)},ci,stats]=resampled_ttest(cv.class_results(:,((cv.M*2)+18)),cv.class_results(:,3),te_tr_ratio,alpha);
        %p-value of accuracy, LDODE vs LDODE utility 80
        [h,tmp{((cv.M*2)+32)},ci,stats]=resampled_ttest(cv.class_results(:,((cv.M*2)+19)),cv.class_results(:,3),te_tr_ratio,alpha);
    end
    
    %mean count trunc imprecise
    tmp(((cv.M*2)+33))=num2cell(nanmean(cv.class_results(:,((cv.M*2)+22)),1));
    %     end
    
    %
    
    
    %GCMA
    tmp(((cv.M*2)+34):((cv.M*2)+42))=num2cell(nanmean(cv.class_results(:,((cv.M*2)+23):((cv.M*2)+31)),1));
    
    if ~isempty(find(isnan(cv.class_results(:,((cv.M*2)+28):((cv.M*2)+29)))==1))
        tmp{((cv.M*2)+43):((cv.M*2)+44)}=num2str(NaN);
    else
        %p-value of accuracy, LDODE vs LDODE utility 65
        [h,tmp{((cv.M*2)+43)},ci,stats]=resampled_ttest(cv.class_results(:,((cv.M*2)+28)),cv.class_results(:,2),te_tr_ratio,alpha);
        %p-value of accuracy, LDODE vs LDODE utility 80
        [h,tmp{((cv.M*2)+44)},ci,stats]=resampled_ttest(cv.class_results(:,((cv.M*2)+29)),cv.class_results(:,2),te_tr_ratio,alpha);
    end
     
    
    %DCMA
    tmp(((cv.M*2)+45):((cv.M*2)+53))=num2cell(nanmean(cv.class_results(:,((cv.M*2)+32):((cv.M*2)+40)),1));
    
    if ~isempty(find(isnan(cv.class_results(:,((cv.M*2)+37):((cv.M*2)+38)))==1))
        tmp{((cv.M*2)+54):((cv.M*2)+55)}=num2str(NaN);
    else
        %p-value of accuracy, LDODE vs LDODE utility 65
        [h,tmp{((cv.M*2)+54)},ci,stats]=resampled_ttest(cv.class_results(:,((cv.M*2)+37)),cv.class_results(:,3),te_tr_ratio,alpha);
        %p-value of accuracy, LDODE vs LDODE utility 80
        [h,tmp{((cv.M*2)+55)},ci,stats]=resampled_ttest(cv.class_results(:,((cv.M*2)+38)),cv.class_results(:,3),te_tr_ratio,alpha);
    end
    
    %percent imprecision 
    
    tmp(((cv.M*2)+56))=num2cell(nanmean(cv.class_results(:,((cv.M*2)+41)),1));
    tmp(((cv.M*2)+57))=num2cell(nanmean(cv.class_results(:,((cv.M*2)+42)),1));       
    
    for j=1:length(tmp)
        fprintf(fid,'%2.5f %s',tmp{j},',');
    end
    
    clear tmp;
    %fprintf(fid,'\n');
    
    fclose(fid);
end
