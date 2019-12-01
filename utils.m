function utils
%%
tmp=dir ('results*.mat');
fid=fopen('dset_list.csv','w');

for i=1:length(tmp)
    current_file=tmp(i).name;
    load (current_file);
    report_ode_results(report_results);
    dset=strrep(current_file,'results_','');
    runs=sum(report_results.class_results(:,1)>0);
    fprintf(fid,'%s %s %d \n',dset, ',',runs);
 end
fclose(fid);


%%
for i=80 
    if sum([imprecise_classes{i}==imprecise_classes_buggy{i}]==0)>0 
        disp(i); 
    end 
end