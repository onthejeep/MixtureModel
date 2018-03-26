import numpy as np;
import sklearn.cluster as cl;
import scipy.stats as sas;


def Optimizer(data, distribution, numComponent):
    if distribution == 'normal':
        return GaussianMixture(data, numComponent);
    elif distribution == 'lognormal':
        return LognormalMixture(data, numComponent);

def GaussianMixture(data, numComponent):
    Mu, Sigma, Weight = InitializeVariable(data, 'normal', numComponent);
    DataLength = len(data);
    Q = np.empty((DataLength, numComponent));
    Iteration = 100;
    Loglikelihood = [0.0] * Iteration;

    for iter in range(Iteration):
        SumDenominator = 0;
        for k in range(numComponent):
            SumDenominator += Weight[k] * NormalPdf(data, Mu[k], Sigma[k]);
        for k in range(numComponent):
            Q[:, k] = Weight[k] * NormalPdf(data, Mu[k], Sigma[k]) / SumDenominator;

        Q[Q < 1e-60] = 1e-15;
        Q[Q > 1 - 1e-60] = 1e-15;

        for k in range(numComponent):
            Weight[k] = np.mean(Q[:, k]);
            Mu[k]     = np.sum(Q[:, k] * data) / np.sum(Q[:, k]);
            Sigma[k]  = np.sqrt( np.sum(Q[:, k] * np.square(data - Mu[k])) / np.sum(Q[:, k]) );

        LogTemp = 0;
        for k in range(numComponent):
            Temp = Weight[k] * sas.norm.pdf(data, Mu[k], Sigma[k]);
            Temp[Temp < 1e-40] = 1e-40;
            LogTemp += np.sum(np.log(Temp));

        Loglikelihood[iter] = LogTemp;

    return Mu, Sigma, Weight, Loglikelihood;

def LognormalMixture(data, numComponent):
    Mu, Sigma, Weight = InitializeVariable(data, 'lognormal', numComponent);
    DataLength = len(data);
    Q = np.empty((DataLength, numComponent));
    Iteration = 100;
    Loglikelihood = [0.0] * Iteration;

    for iter in range(Iteration):
        SumDenominator = 0;
        for k in range(numComponent):
            SumDenominator += Weight[k] * LognormalPdf(data, Mu[k], Sigma[k]);
        for k in range(numComponent):
            Q[:, k] = Weight[k] * LognormalPdf(data, Mu[k], Sigma[k]) / SumDenominator;

        Q[Q < 1e-60] = 1e-15;
        Q[Q > 1 - 1e-60] = 1e-15;

        for k in range(numComponent):
            Weight[k] = np.mean(Q[:, k]);
            Mu[k]     = np.sum(Q[:, k] * np.log(data)) / np.sum(Q[:, k]);
            Sigma[k]  = np.sqrt( np.sum(Q[:, k] * np.square(np.log(data) - Mu[k])) / np.sum(Q[:, k]) );

        LogTemp = 0;
        for k in range(numComponent):
            Temp = Weight[k] * LognormalPdf(data, Mu[k], Sigma[k]);
            Temp[Temp < 1e-20] = 1e-20;
            LogTemp += np.sum(np.log(Temp));

        Loglikelihood[iter] = LogTemp;

    return Mu, Sigma, Weight, Loglikelihood;

def NormalPdf(data, mu, sigma):
    Exponential = -1 * np.square(data - mu) / (2 * np.square(sigma));
    Probability = 1 / (np.sqrt(2 * np.pi * np.square(sigma))) * np.exp(Exponential);
    return Probability;

def LognormalPdf(data, mu, sigma):
    Exponential = - np.square(np.log(data) - mu) / (2 * np.square(sigma));
    Probability = 1 / (np.sqrt(2 * np.pi) * sigma * data) * np.exp(Exponential);
    return Probability;

def SingleDistribution_Mean(distribution, mu, sigma):
    if distribution == 'normal':
        return mu;
    elif distribution == 'lognormal':
        return np.exp(mu + np.square(sigma) / 2);

def SingleDistribution_Variance(distribution, mu, sigma):
    if distribution == 'normal':
        return np.square(sigma);
    elif distribution == 'lognormal':
        return (np.exp(np.square(sigma)) - 1) * np.exp(2 * np.array(mu) + np.square(sigma));

def SingleDistribution_Std(distribution, mu, sigma):
    if distribution == 'normal':
        return sigma;
    elif distribution == 'lognormal':
        Variance = SingleDistribution_Variance(distribution, mu, sigma);
        return np.sqrt(Variance);

def SingleDistribution_Skewness(distribution, sigma):
    if distribution == 'normal':
        return np.zeros(sigma.shape);
    elif distribution == 'lognormal':
        return (np.exp(np.square(sigma)) + 2) * np.sqrt(np.exp(np.square(sigma)) - 1);

def InitializeVariable(data, distribution, numComponent):
    kmeans = cl.KMeans(n_clusters = numComponent, precompute_distances = True, algorithm= 'full');
    ReshapedData = np.reshape(data, (-1, 1));
    kmeans = kmeans.fit(ReshapedData);
    Centroids = kmeans.cluster_centers_;
    Labels = kmeans.labels_;

    Mu = [0] * numComponent;        
    Sigma = [0] * numComponent;        
    Weight = [0] * numComponent;

    if distribution == 'normal':
        for i in range(numComponent):
            ClusteredData = data[Labels == i];
            Mu[i] = np.mean(ClusteredData);
            Sigma[i] = np.std(ClusteredData);
            Weight[i] = len(ClusteredData) / len(data);
    elif distribution == 'lognormal':
        for i in range(numComponent):
            ClusteredData = data[Labels == i];
            SampleMean = np.mean(ClusteredData);
            SampleVar = np.var(ClusteredData);

            Var = np.log(SampleVar * np.exp(-2 * np.log(SampleMean)) + 1);
            Sigma[i] = np.sqrt(Var);
            Mu[i] = np.log(SampleMean) - Var / 2;
            Weight[i] = len(ClusteredData) / len(data);

    return Mu, Sigma, Weight;