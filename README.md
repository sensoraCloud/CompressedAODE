# CompressedAODE

Compressed-AODE bayesian probabilistic imprecise classifier

## Description

We propose the COMP-AODE classifier, which adopts
the compression-based approach [1] to average the posterior probabilities computed by different non-naive classifiers (SPODEs).
COMP-AODE improves classification performance over the wellknown AODE model. COMP-AODE assumes a uniform prior
over the SPODEs; we then develop the credal classifier COMPAODE*, substituting the uniform prior by a set of priors. COMPAODE* returns more classes when the classification is priordependent, namely if the most probable class varies with the prior
adopted over the SPODEs. COMP-AODE* achieves higher classification utility than both COMP-AODE and AODE.

## Ref. Publication

Compression-based AODE classifiers  
Corani, G., Antonucci, A., De Rosa, R  
Proc. 20th European Conference on Artificial Intelligence (ECAI 2012), pp. 264–269  
