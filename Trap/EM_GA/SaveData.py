'''save data (Traveltime)'''


import numpy as np;


Mu = [4, 7, 10];
Sigma = [1, 2, 3];
Weight = [0.6, 0.3, 0.1];
DataSize = 1000;
NumComponent = 3;

def FakeData():
    '''simulated data'''
    Traveltime = [];
    for i in range(NumComponent):
        Traveltime = np.append(Traveltime, np.random.normal(Mu[i], Sigma[i], int(DataSize * Weight[i])));
    Traveltime = Traveltime[Traveltime > 0];
    return Traveltime;


# ground truth statistics
# mean = 0.6 * 4 + 0.3 * 7 + 0.1 * 10 = 5.5;
# mu_i - mu = [-1.5, 1.5, 4.5];
# variance = 0.6 * (1.5^2 + 1^2) + 0.3 * (1.5^2 + 2^2) + 0.1 * (1.5^2 + 3^2) = 4.95
# skewness = 0.6 * (-1.5^3 - 3*1.5*1^2) + 0.3 * (1.5^3 + 3 * 1.5 * 2^2) + 0.1 * (4.5^3 + 3*4.5*3^2) = 22.95