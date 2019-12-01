function [ determ,sing_acc,ind_out_size,set_acc,dacc,util_65,util_80,precise_single,precise_set, imprecise_index ] = imprecise_performances( imprecise_classes , true_classes , precise_classes )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%initialiazion, for each indicator a vector as long as the test set
size_test=size(true_classes,1);

determ=0;
ind_out_size=0;
sing_acc=0;
set_acc=0;

dacc=zeros(1,size_test);
util_65=zeros(1,size_test);
util_80=zeros(1,size_test);
count_set=0;

precise_single=0;
precise_set=0;

imprecise_index=zeros(size_test,1);

for d=1:size_test
   
    size_out=size(imprecise_classes{d,1},2);
    
    
    accurate=0;
    
    if size_out==1
        determ=determ+1;
        if imprecise_classes{d,1}==true_classes(d,1)
            sing_acc=sing_acc+1;
            accurate=1;
        end
        
        if precise_classes(d,1)==true_classes(d,1)
            precise_single=precise_single+1;            
        end
        
        if imprecise_classes{d,1}~=precise_classes(d,1)
            beep
              warning('Single imprecise different precise classification!!');
        end
        
    else
        imprecise_index(d,1)=1;
        count_set=count_set+1;
        ind_out_size=ind_out_size+size_out;
        
        if sum(ismember(true_classes(d,1),imprecise_classes{d,1}))>0
            set_acc=set_acc+1;
            accurate=1;
        end
        
        if precise_classes(d,1)==true_classes(d,1)
            precise_set=precise_set+1;            
        end
        
    end
    
    dacc(1,d)=accurate/size_out;
    util_65(1,d)=util65(dacc(1,d));
    util_80(1,d)=util80(dacc(1,d));
    
    
end

if determ==0
    sing_acc=NaN;
    precise_single=NaN;
else
    sing_acc=sing_acc/determ;
    precise_single=precise_single/determ;
end

if count_set==0
    precise_set=NaN;
    set_acc=NaN;
else
   precise_set=precise_set/count_set;
   set_acc=set_acc/count_set;
   ind_out_size=ind_out_size/count_set;
end

if ind_out_size==0
    ind_out_size=NaN;
end

determ=determ/size_test;

dacc=mean(dacc);

util_65=mean(util_65);

util_80=mean(util_80);


function util=util80(x)
util=-1.2*x^2 + 2.2*x;
end

function util=util65(x)
util=-0.6*x^2 + 1.6*x;
end


end

