import numpy as np;
import scipy.stats as sas;
import statsmodels.distributions;
import ReadData;
import matplotlib.pyplot as plt;


def OnesampleTest(data, replication = 1000, sigificance = 0.05):
    Theta_b = np.repeat(0.0, replication);
    Theta = np.mean(data);

    DataSize = len(data);
    for i in range(replication):
        SampleData = np.random.choice(data, DataSize, True);
        Theta_b[i] = np.mean(SampleData);

    EmpiricalCDF = statsmodels.distributions.ECDF(Theta_b);

    LowerBound = np.percentile(Theta_b, (sigificance / 2) * 100);
    UpperBound = np.percentile(Theta_b, (1 - sigificance / 2) * 100);

    if Theta > LowerBound and Theta < UpperBound:
        Test = 'not reject';
    else:
        Test = 'reject';

    p_value = (1 - EmpiricalCDF(Theta)) * 2;

    return Theta_b, LowerBound, UpperBound, p_value, Test;

# def 

def R_Summary(data):
    return np.min(data), np.max(data), np.mean(data), np.std(data), \
        np.percentile(data, 25), np.percentile(data, 50), np.percentile(data, 75);


if __name__ == '__main__':
    EntireParameters = ReadData.LoadCSV('Tensorflow/Result/result_meanvar_normal.csv');
    Observations = EntireParameters.as_matrix(['mm_skew']).flatten();
    Observations = Observations[~np.isnan(Observations)];

    # 5.5, 4.95, 22.95
    TheoreticalValue = 22.95;
    Theta_b, LowerBound, UpperBound, p_value, Test = OnesampleTest(Observations, replication = 1000, sigificance = 0.05);
    print(LowerBound, UpperBound, p_value, Test);

    print(R_Summary(Observations));

    plt.hist(Theta_b, bins=40, density=True, lw=3, fc=(0, 0, 0, 0.5));
    plt.axvline(x = LowerBound, color = 'g');
    plt.axvline(x = UpperBound, color = 'g');
    plt.axvline(x = TheoreticalValue, color = 'r');
    plt.show();