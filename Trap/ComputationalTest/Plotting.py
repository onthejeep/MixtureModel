import pandas as pan;
import numpy as np;
import matplotlib.pyplot as plt;
import MixtureModel;


def PlotDistributionReplication(filePath):
    MixtureParameters = pan.read_csv(filePath, sep= ',', header = 0);
    NumComponent = 3;
    for i in range(MixtureParameters.shape[0]):
        Mu = np.array(MixtureParameters.iloc[i, 0:NumComponent]);
        Sigma = np.sqrt(MixtureParameters.iloc[i, NumComponent:NumComponent*2]);
        Weight = np.array(MixtureParameters.iloc[i, NumComponent*2:NumComponent*3]);

        MinValue, MaxValue = 0, 20;
        X, Y = MixtureModel.MixturePDF(MinValue, MaxValue, 'normal', NumComponent, Mu, Sigma, Weight);

        plt.plot(X, Y, 'g');

    plt.show();


PlotDistributionReplication('EM/Result/result_musigma_normal.csv');