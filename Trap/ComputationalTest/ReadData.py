import pandas as pan;
import numpy as np;


def LoadCSV(filePath):
    MixtureParameters = pan.read_csv(filePath, sep= ',', header = 0);
    # ['mean1', 'mean2', 'mean3', 'var1', 'var2', 'var3', 'sk1', 'sk2', 'sk3', 'w1', 'w2', 'w3', 'MM_Mean', 'MM_Var', 'MM_Skew']
    #print(MixtureParameters.columns);
    return MixtureParameters;

def LoadCSV_wParameter(filePath, columnName = 'mean1'):
    MixtureParameters = pan.read_csv(filePath, sep= ',', header = 0);
    return MixtureParameters.as_matrix([columnName]);

# def LoadMixtureModelParamters(filePath):
#     MixtureParameters = pan.read_csv(filePath, sep= ',', header = 0);
#     print(np.array(MixtureParameters.iloc[0, 0:3]));
#     print(MixtureParameters.iloc[0, 3:6]);
#     print(MixtureParameters.iloc[0, 6:9]);

# LoadMixtureModelParamters('EM/Result/Result_normal.csv');

