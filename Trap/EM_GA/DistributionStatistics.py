'''
calcualte statistics from mixture distribtuions
'''

import numpy as np;

def MixtureModel_Mean(mean, weight):
    ''' calculate mean of mixture distributions'''
    return np.sum(mean * weight);

def MixtureModel_Var(mean, var, weight):
    '''calculate variance of mixture distributions'''
    Mean = MixtureModel_Mean(mean, weight);
    Result = np.sum(weight * (np.square(mean - Mean) + var));
    return Result;

def MixtureModel_Skewness(mean, var, skewness, weight):
    '''calculate skewness of mixture distributions'''
    Mean = MixtureModel_Mean(mean, weight);
    Result = np.sum(weight * (np.power(mean - Mean, 3) + 3 * (mean - Mean) * var + skewness));
    return Result;

def Empirical_Skewness(data):
    '''calcualte empirical skewness'''
    SampleMean = np.mean(data);
    M3 = np.mean(np.power(data - SampleMean, 3));
    S3 = np.power(np.std(data), 3);
    Skewness = M3 / S3;
    return M3;