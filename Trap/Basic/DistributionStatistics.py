'''
calcualte statistics from mixture distribtuions
'''

import numpy as np;

def MixtureModel_Mean(mu, weight, k):
    ''' calculate mean of mixture distributions'''
    Result = 0;
    for i in range(k):
        Result += mu[i] * weight[i];
    return Result;

def MixtureModel_Var(mu, var, weight, k):
    '''calculate variance of mixture distributions'''
    Mean = MixtureModel_Mean(mu, weight, k);
    Result = 0;
    for i in range(k):
        Result += weight[i] * (np.square(mu[i] - Mean) + var[i]);
    return Result;

def MixtureModel_Skewness(mu, var, skewness, weight, k):
    '''calculate skewness of mixture distributions'''
    Mean = MixtureModel_Mean(mu, weight, k);
    Result = 0;
    for i in range(k):
        Result += weight[i] * (np.power(mu[i] - Mean, 3) + 3 * (mu[i] - Mean) * var[i] + skewness[i]);
    return Result;

def Empirical_Skewness(data):
    '''calcualte empirical skewness'''
    SampleMean = np.mean(data);
    M3 = np.mean(np.power(data - SampleMean, 3));
    S3 = np.power(np.std(data), 3);
    Skewness = M3 / S3;
    return M3;