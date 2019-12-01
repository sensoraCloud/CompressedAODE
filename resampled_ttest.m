function [h,p,ci,stats] = resampled_ttest(x,y,te_tr_ratio,alpha)
%   [h,p,ci,stats] = TTEST(x,y,te_tr_ratio, alpha) performs a resampling-corrected paired T-test,
%where te_tr_ratio is the ratio of instances in test and training sets; for cross-validation, this is 1/(nfolds-1).


if ~isequal(size(y),size(x))
        error('stats:ttest:InputSizeMismatch',...
              'The data in a paired t-test must be the same size.');
end
x = x - y;
m=0;
if nargin < 5 || isempty(alpha)
    alpha = 0.05;
elseif ~isscalar(alpha) || alpha <= 0 || alpha >= 1
    error('stats:ttest:BadAlpha','ALPHA must be a scalar between 0 and 1.');
end


if (te_tr_ratio>1)
    error('te_tr_ratio should be <= 1');
end


tail = 0;

if nargin < 5 || isempty(dim)
    % Figure out which dimension mean will work along
    dim = find(size(x) ~= 1, 1);
    if isempty(dim), dim = 1; end
end

nans = isnan(x);
if any(nans(:))
    samplesize = sum(~nans,dim);
else
    samplesize = size(x,dim); % a scalar, => a scalar call to tinv
end
df = max(samplesize - 1,0);
xmean = nanmean(x,dim);
sdpop = nanstd(x,[],dim);
%GC
%ser = sdpop ./ sqrt(samplesize);
ser = sdpop .* sqrt(1/samplesize+te_tr_ratio);
tval = (xmean - m) ./ ser;
if nargout > 3
    stats = struct('tstat', tval, 'df', cast(df,class(tval)), 'sd', sdpop);
    if isscalar(df) && ~isscalar(tval)
        stats.df = repmat(stats.df,size(tval));
    end
end

% Compute the correct p-value for the test, and confidence intervals
% if requested.
if tail == 0 % two-tailed test
    p = 2 * tcdf(-abs(tval), df);
    if nargout > 2
        crit = tinv((1 - alpha / 2), df) .* ser;
        ci = cat(dim, xmean - crit, xmean + crit);
    end
elseif tail == 1 % right one-tailed test
    p = tcdf(-tval, df);
    if nargout > 2
        crit = tinv(1 - alpha, df) .* ser;
        ci = cat(dim, xmean - crit, Inf(size(p)));
    end
elseif tail == -1 % left one-tailed test
    p = tcdf(tval, df);
    if nargout > 2
        crit = tinv(1 - alpha, df) .* ser;
        ci = cat(dim, -Inf(size(p)), xmean + crit);
    end
else
    error('stats:ttest:BadTail',...
          'TAIL must be ''both'', ''right'', or ''left'', or 0, 1, or -1.');
end
% Determine if the actual significance exceeds the desired significance
h = cast(p <= alpha, class(p));
h(isnan(p)) = NaN; % p==NaN => neither <= alpha nor > alpha
