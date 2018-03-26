import numpy as np;
import sklearn.cluster as cl;
import scipy.stats as sas;



def Loglikelihood(data, numComponent, mu, sigma, weight):
    Loglikelihood = 0;
    for k in range(numComponent - 1):
        Temp = weight[k] * NormalPdf(data, mu[k], sigma[k]);
        Temp[Temp < 1e-20] = 1e-20;
        Loglikelihood += np.sum(np.log(Temp));
    
    Temp = weight[-1] * LognormalPdf(data, mu[-1], sigma[-1]);
    Temp[Temp < 1e-20] = 1e-20;
    Loglikelihood += np.sum(np.log(Temp));

    return Loglikelihood;

def MixturePDF(data, numComponent, mu, sigma, weight):
    NumPoint = 100;
    X = np.linspace(np.min(data), np.max(data), num= NumPoint);
    Y = [0] * NumPoint;
    for i in range(numComponent - 1):
        Y += weight[i] * NormalPdf(X, mu[i], sigma[i]);
    Y += weight[-1] * LognormalPdf(X, mu[-1], sigma[-1]);
    return X, Y;
        

def NormalPdf(data, mu, sigma):
    Exponential = -1 * np.square(data - mu) / (2 * np.square(sigma));
    Probability = 1 / (np.sqrt(2 * np.pi * np.square(sigma))) * np.exp(Exponential);
    return Probability;

def LognormalPdf(data, mu, sigma):
    Exponential = - np.square(np.log(data) - mu) / (2 * np.square(sigma));
    Probability = 1 / (np.sqrt(2 * np.pi) * sigma * data) * np.exp(Exponential);
    return Probability;

def SingleDistribution_Mean(numComponent, mu, sigma):
    return np.append(mu[0:(numComponent - 1)], np.exp(mu[-1] + np.square(sigma[-1]) / 2));

def SingleDistribution_Variance(numComponent, mu, sigma):    
    return np.append(np.square(sigma[0:(numComponent - 1)]), 
        (np.exp(np.square(sigma[-1])) - 1) * np.exp(2 * np.array(mu[-1]) + np.square(sigma[-1])));

def SingleDistribution_Std(numComponent, mu, sigma):
    Variance = SingleDistribution_Variance(numComponent, mu, sigma);
    return np.sqrt(Variance);

def SingleDistribution_Skewness(numComponent, sigma):
    return np.append(np.repeat(0, numComponent - 1), 
        (np.exp(np.square(sigma[-1])) + 2) * np.sqrt(np.exp(np.square(sigma[-1])) - 1));

def InitializeVariable(data, numComponent):
    kmeans = cl.KMeans(n_clusters = numComponent, precompute_distances = True, algorithm= 'full');
    ReshapedData = np.reshape(data, (-1, 1));
    kmeans = kmeans.fit(ReshapedData);
    Centroids = kmeans.cluster_centers_;
    Labels = kmeans.labels_;

    Mu = [0] * numComponent;        
    Sigma = [0] * numComponent;        
    Weight = [0] * numComponent;

    for i in range(numComponent - 1):
        ClusteredData = data[Labels == i];
        Mu[i] = np.mean(ClusteredData);
        Sigma[i] = np.std(ClusteredData);
        Weight[i] = len(ClusteredData) / len(data);
 
    ClusteredData = data[Labels == (numComponent - 1)];
    SampleMean = np.mean(ClusteredData);
    SampleVar = np.var(ClusteredData);

    Var = np.log(SampleVar * np.exp(-2 * np.log(SampleMean)) + 1);
    Sigma[-1] = np.sqrt(Var);
    Mu[-1] = np.log(SampleMean) - Var / 2;
    Weight[-1] = len(ClusteredData) / len(data);

    return Mu, Sigma, Weight;