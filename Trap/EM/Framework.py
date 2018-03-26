import SaveData;
import DistributionStatistics as DistributionStatistics;
import matplotlib.pyplot as plt;
import numpy as np;
import MixtureModel as MixtureModel;
import csv;
import math;


def ExecuteTraining(dataSize):
    Distribution = 'normal';
    NumComponent = 3;

    TraveltimeData = np.loadtxt('Basic/Result/Traveltime_{}.txt'.format(dataSize));
    NumReplication = TraveltimeData.shape[0];

    MM_Mean = np.repeat(0.0, NumReplication);
    MM_Var = np.repeat(0.0, NumReplication);
    MM_Skew = np.repeat(0.0, NumReplication);

    MeanVarFile = open('EM/Result/result_meanvar_{}_{}.csv'.format(Distribution, dataSize), 'w+', newline = '');
    MuSigmaFile = open('EM/Result/result_musigma_{}_{}.csv'.format(Distribution, dataSize), 'w+', newline = '');

    MeanVarReader = csv.writer(MeanVarFile, delimiter = ',');
    MeanVarReader.writerow(['mean1', 'mean2', 'mean3', 'var1', 'var2', 'var3', 'sk1', 'sk2', 'sk3', 'w1', 'w2', 'w3', 'mm_mean', 'mm_var', 'mm_skew']);
    MuSigmaReader = csv.writer(MuSigmaFile, delimiter = ',');
    MuSigmaReader.writerow(['mu1', 'mu2', 'mu3', 'sigma1', 'sigma2', 'sigma3', 'w1', 'w2', 'w3']);

    for i in range(NumReplication):
        if i % (NumReplication / 10) == 0:
            print('Replication = ', i);
        
        Traveltime = TraveltimeData[i];
        Traveltime = Traveltime[Traveltime > 0];

        Mu, Sigma, Weight, Loglikelihood = MixtureModel.Optimizer(Traveltime, Distribution, NumComponent);
        Mu, Sigma, Weight, Loglikelihood = np.array(Mu), np.array(Sigma), np.array(Weight), np.array(Loglikelihood);

        ReturnIndex = np.argsort(Mu);
        Mu = Mu[ReturnIndex];
        Sigma = Sigma[ReturnIndex];
        Weight = Weight[ReturnIndex];

        MuSigmaReader.writerow([Mu[0], Mu[1], Mu[2], Sigma[0], Sigma[1], Sigma[2], \
            Weight[0], Weight[1], Weight[2]]);

        Mean, Variance, Skewness = MixtureModel.SingleDistribution_Mean(Distribution, Mu, Sigma),\
                MixtureModel.SingleDistribution_Variance(Distribution, Mu, Sigma),\
                MixtureModel.SingleDistribution_Skewness(Distribution, Sigma);

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
            MeanVarReader.writerow([Mean[0], Mean[1], Mean[2], Variance[0], Variance[1], Variance[2], \
                Skewness[0], Skewness[1], Skewness[2], \
                Weight[0], Weight[1], Weight[2], \
                MM_Mean[i], MM_Var[i], MM_Skew[i]]);
        else:
            MeanVarReader.writerow([Mean[0], Mean[1], Mean[2], Variance[0], Variance[1], Variance[2], \
                Skewness[0], Skewness[1], Skewness[2], \
                Weight[0], Weight[1], Weight[2], \
                MM_Mean[i], MM_Var[i], MM_Skew[i]]);


    MeanVarFile.close();
    MuSigmaFile.close();


        

        

if __name__ == '__main__':
    ExecuteTraining(100);
    ExecuteTraining(1000);