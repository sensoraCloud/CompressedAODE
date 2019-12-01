function myfriedman(filename)
data = csvread(filename);           % Read the CSV file
[d,c] = size(data);                 % Read the number of datasets (d, rows) e classifiers (c, columns)
c_act = ones(1,c);                  % By default all the classifiers are compared
%u65
%c_act = [1 0 1 1 0 0 0 0 0 0];     % Optionally an array denoting the classifiers (columns) to be compared
%including CDT
c_act = [0 1 1 1 0 0 0 0 0 0];     % Optionally an array denoting the classifiers (columns) to be compared
%u80
%c_act = [0 0 0 0 1 0 1 1 0 0];      % can be specified (NCC, CMA, COMPRESS) 
%c_act = [0 0 0 0 0 1 1 1 0 0];      % can be specified (CDT, CMA, COMPRESS) 
k = sum(c_act);                     % Number of classifiers to be compared
ranks = zeros(d,k);                 % Initialize the matrix with the ranks for the d datasets
av_ranks = zeros(1,k);              % Initialize the array with the average ranks
% Fill the rows of ranks with the FractionalRanking
for i = 1:d,
      ranks(i,:) = k+1-FractionalRankings(data(i,c_act==1));
end
for i = 1:k
     av_ranks(i) = sum(ranks(:,i))/d;    
end
% Test statistic for the Friedman test
% For k,n large enough should obey chisquare distribution
% with (k-1) degrees of freedom
squares = av_ranks.^2;
friedman = 12*d/(k*(k+1))*(sum(squares)-(k*(k+1)^2)/4);
pvalue_friedman = 1-cdf('chi2',friedman,k-1);
 
% Test statistic according to Iman and Davenport
% This is distributed as a F-distribution
% with (k-1) and (k-1)(d-1) degrees of freedom
imandavenport = (d-1)*friedman/(d*(k-1)-friedman);
pvalue_imandavenport = 1-cdf('f',imandavenport,k-1,(k-1)*(d-1));

% Critical values for post-hoc Nemenyi test
q_5 = [0,1.960,2.343,2.569,2.728,2.850,2.949,3.031,3.102,3.164];
q_10 = [0,1.645,2.052,2.291,2.459,2.589,2.693,2.780,2.855,2.920];
CD5 = q_5(k) * sqrt(k*(k+1)/(6*d));
CD10 = q_10(k) * sqrt(k*(k+1)/(6*d));

% Comparisons
comparisons = zeros(k);
significance5 = comparisons;
significance10 = significance5;
fprintf('Friedman p-value = %2.5f\n',pvalue_friedman);
fprintf('Iman Davenport p-value = %2.5f\n',pvalue_imandavenport);
fprintf('CD5 = %2.3f \nCD10 = %2.3f\n',CD5,CD10);
for i = 1:k
      for j = 1:k
          if j>i
          comparisons(i,j) = av_ranks(i)- av_ranks(j);
          if abs(comparisons(i,j))>CD5
              significance5(i,j) = 1;
          end
          if abs(comparisons(i,j))>CD10
              significance10(i,j) = 1;
          end          
          end
      end
end
str = '';
for i = 1:k
      str = strcat(str,sprintf('  (%2.6f,%i) ',av_ranks(i),i));
end
comparisons
significance5
significance10
disp(str);
end