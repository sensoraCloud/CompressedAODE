function [ class_probs ] = get_class_prob(bnet_engine, istance )
%get_class_prob(net, istance ) return probability class from net
%INPUT  bnet_engine         bayesian network engine (bnet_engine = jtree_inf_engine(bnet))
%       istance     [{class} {a1} {a2} .. {ak}] sample to classfy
%OUTPUT class_prob  P(c|net,istance)

%engine = jtree_inf_engine(bnet);
evidence=istance;
%remove class
evidence{1} = [];
[bnet_engine, ll] = enter_evidence(bnet_engine, evidence);
marg = marginal_nodes(bnet_engine, 1);
class_probs=marg.T;
class_probs(class_probs<0.0001)=0.0001;
class_probs=class_probs./sum(class_probs);

if (sum(class_probs)==0)
    disp('warning! all classes have prob=0');
end

end

