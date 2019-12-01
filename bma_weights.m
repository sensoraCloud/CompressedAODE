function [ exp_weights, high_lliks_idx ] = bma_weights( lliks )
%[ exp_weights, high_lliks_idx ] = bma_weights( lliks )
%code is robustified in order to manage exceptional cases in which even the
%difference is not exponentiable
%low_lliks_idx contains all the models whose liks are smaller by a factor 10e4
%then the model with maximal likelihood, namely those whose lliks are at a
%distance of less than log(10^4)=9.21 from the model with maximal likelihood.
%

first=true;
good_idx=1:length(lliks);
bad_idx=[];

while (first || ~isempty(isinf_idx))
    isinf_idx=[];
    shifted_lliks=[];
    exp_ll=[];
    first=false;
    [shift,min_idx]=min(lliks(good_idx));
    shifted_lliks(good_idx)=lliks(good_idx)-shift;
    exp_ll(good_idx)=exp(shifted_lliks(good_idx));
    isinf_idx=find(isinf(exp_ll));
    if ~isempty(isinf_idx) 
        %bad_idx=[bad_idx find(lliks==min(lliks(good_idx)))];
        bad_idx=[bad_idx find(lliks==shift)];
    end
    good_idx=setdiff(good_idx,bad_idx);
end

total=sum(exp_ll(good_idx));
exp_weights(good_idx)=exp_ll(good_idx)/total;
epsilon=0.000001;
exp_weights(bad_idx)=epsilon;
exp_weights(exp_weights<epsilon)=epsilon;
tmp=sum(exp_weights);
exp_weights=exp_weights/tmp;

max_ll=max(lliks);
high_lliks_idx=find(lliks+log(10^4)>max_ll); 
%OLD CODE
% shift=min(lliks);
% ll=lliks-shift;
% ll=exp(ll);
% total=sum(ll);
% exp_weights=ll/total;

end

