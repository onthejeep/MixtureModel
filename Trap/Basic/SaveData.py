'''save data (Traveltime)'''

import numpy as np;



def FakeData(dataSize):
    '''simulated data'''
    Mu = [4, 7, 10];
    Sigma = [1, 2, 3];
    Weight = [0.6, 0.3, 0.1];
    NumComponent = 3;

    Traveltime = [];
    for i in range(NumComponent):
        Traveltime = np.append(Traveltime, np.random.normal(Mu[i], Sigma[i], np.int(dataSize * Weight[i])));
    # Traveltime = Traveltime[Traveltime > 0];
    return Traveltime;


# ground truth statistics
# mean = 0.6 * 4 + 0.3 * 7 + 0.1 * 10 = 5.5;
# mu_i - mu = [-1.5, 1.5, 4.5];
# variance = 0.6 * (1.5^2 + 1^2) + 0.3 * (1.5^2 + 2^2) + 0.1 * (1.5^2 + 3^2) = 4.95
# skewness = 0.6 * (-1.5^3 - 3*1.5*1^2) + 0.3 * (1.5^3 + 3 * 1.5 * 2^2) + 0.1 * (4.5^3 + 3*4.5*3^2) = 22.95

if __name__ == '__main__':
    NumDataset = 1000;
    DataSize = 1000;
    Traveltime = [np.ndarray] * NumDataset;
    for i in range(NumDataset):
        np.random.seed(i);
        Traveltime[i] = FakeData(DataSize);

    Traveltime = np.array(Traveltime);
    np.savetxt('Basic/Result/Traveltime_{}.txt'.format(DataSize), Traveltime, fmt = '%.4e');

    # AA = np.loadtxt('Basic/Result/Traveltime.txt');
    # print(np.mean(AA[0]));