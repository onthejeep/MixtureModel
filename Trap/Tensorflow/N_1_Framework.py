import N_1_MixtureModel;
import SaveData;
import matplotlib.pyplot as plt;
import numpy as np;
import DistributionStatistics;
import csv;
import N_1_Deeplearning;
import math;
import datetime;




def ExecuteTraining(dataSize):

    TraveltimeData = np.loadtxt('Basic/Result/Traveltime_{}.txt'.format(dataSize));
    NumReplication = TraveltimeData.shape[0];

    MM_Mean = [0.0] * NumReplication;
    MM_Var = [0.0] * NumReplication;
    MM_Skew = [0.0] * NumReplication;

    Optimizer = N_1_Deeplearning.MLE();
    Optimizer._Distribution = 'normal';
    Optimizer._NumComponent = 2;
    Optimizer.DefineFlow();

    print(datetime.datetime.now());

    MeanVarFile = open('Tensorflow/Result/n_1_result_meanvar_{}_{}.csv'.format(Optimizer._Distribution, dataSize), 'w+', newline = '');
    MuSigmaFile = open('Tensorflow/Result/n_1_result_musigma_{}_{}.csv'.format(Optimizer._Distribution, dataSize), 'w+', newline = '');

    MeanVarReader = csv.writer(MeanVarFile, delimiter = ',');
    MeanVarReader.writerow(['mean1', 'mean2', 'var1', 'var2', 'sk1', 'sk2', 'w1', 'w2', 'mm_mean', 'mm_var', 'mm_skew']);
    MuSigmaReader = csv.writer(MuSigmaFile, delimiter = ',');
    MuSigmaReader.writerow(['mu1', 'mu2', 'sigma1', 'sigma2', 'w1', 'w2']);

    for i in range(NumReplication):
        if i % (NumReplication / 10) == 0:
            print('Replication = ', i);
        
        Traveltime = TraveltimeData[i];
        Traveltime = Traveltime[Traveltime > 0];

        Mu, Sigma, Weight, Loglikelihood = Optimizer.Training(data = Traveltime);
        Mu, Sigma, Weight, Loglikelihood = np.array(Mu), np.array(Sigma), np.array(Weight), np.array(Loglikelihood);
        
        ReturnIndex = np.argsort(-Weight);
        Mu = Mu[ReturnIndex];
        Sigma = Sigma[ReturnIndex];
        Weight = Weight[ReturnIndex];

        MuSigmaReader.writerow([Mu[0], Mu[1], Sigma[0], Sigma[1], \
            Weight[0], Weight[1]]);
        
        Mean, Variance, Skewness = N_1_MixtureModel.SingleDistribution_Mean(Optimizer._NumComponent, Mu, Sigma),\
            N_1_MixtureModel.SingleDistribution_Variance(Optimizer._NumComponent, Mu, Sigma),\
            N_1_MixtureModel.SingleDistribution_Skewness(Optimizer._NumComponent, Sigma);

        ReturnIndex = np.argsort(Mean);
        Mean = Mean[ReturnIndex];
        Variance = Variance[ReturnIndex];
        Skewness = Skewness[ReturnIndex];
        Weight = Weight[ReturnIndex];

        MM_Mean[i] = DistributionStatistics.MixtureModel_Mean(Mean, Weight);
        MM_Var[i] = DistributionStatistics.MixtureModel_Var(Mean, Variance, Weight);
        MM_Skew[i] = DistributionStatistics.MixtureModel_Skewness(Mean, Variance, Skewness, Weight);

        if math.isnan(MM_Var[i]):
            print('variance = Nan, seed = ', i);
            MeanVarReader.writerow([Mean[0], Mean[1], Variance[0], Variance[1],  \
                Skewness[0], Skewness[1],  \
                Weight[0], Weight[1], \
                MM_Mean[i], MM_Var[i], MM_Skew[i]]);
        else:
            MeanVarReader.writerow([Mean[0], Mean[1], Variance[0], Variance[1],  \
                Skewness[0], Skewness[1],  \
                Weight[0], Weight[1], \
                MM_Mean[i], MM_Var[i], MM_Skew[i]]);

    MeanVarFile.close();
    MuSigmaFile.close();

    print(datetime.datetime.now());

if __name__ == '__main__':
    ExecuteTraining(100);
    ExecuteTraining(1000);