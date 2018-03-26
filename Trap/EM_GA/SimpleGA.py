import sys;
from math import sin, cos, radians;
import numpy as np;
import matplotlib.pyplot as plt;
import MixtureModel;
import scipy.stats as sas;



class GA_EM:

    """ 
    objective function: minimize -maxlikelihood
    mixture model = Normal_i(mu_i, sigma_i)
    two major operations
        mutation: Mutate(self, crossedSolutions)
        crossover: Crossover(self, selectedSolutions)
    """
    
    def __init__(self):
        self._Distribution = 'normal';
        self._NumComponent = 3;
        self._NumSolution = 400;
        self._MutationRate = 0.1;
        self._CrossoverRate = 0.1;
        self._NumElitism = 2;


    def InitiallizeGeneration(self, data):
        """ 3 * self._NumComponent variables in the objective function"""
        Generation = np.empty((self._NumSolution, self._NumComponent * 3));

        SampleMean = np.mean(data);
        SampleVar  = np.var(data);

        Mu, Sigma, Weight = MixtureModel.InitializeVariable(data, self._Distribution, self._NumComponent);
        Generation[:, 0:self._NumComponent] = np.random.normal(Mu, 0.3, (self._NumSolution, self._NumComponent));
        Generation[:, self._NumComponent : self._NumComponent * 2] = np.random.normal(Sigma, 0.3, (self._NumSolution, self._NumComponent));
        Generation[:, self._NumComponent * 2 : self._NumComponent * 3] = np.tile(Weight, (self._NumSolution, 1));
        Generation[Generation < 0] = SampleMean;
        return Generation;

    def PerformEM(self, data, mu, sigma, weight):
        '''
        mu, sigma, and weight are shallow copied, any changed on the three variables in this function will 
        be reflected in the original variables(memory)
        no return function is required
        '''
        DataLength = len(data);
        Q = np.empty((DataLength, self._NumComponent));
        Loglikelihood = 0;

        if self._Distribution == 'normal':
            DistributionPDF = MixtureModel.NormalPdf;
        elif self._Distribution == 'lognormal':
            DistributionPDF = MixtureModel.LognormalPdf;

        SumDenominator = 0;
        for k in range(self._NumComponent):
            SumDenominator += weight[k] * DistributionPDF(data, mu[k], sigma[k]);
        for k in range(self._NumComponent):
            Q[:, k] = weight[k] * DistributionPDF(data, mu[k], sigma[k]) / SumDenominator;

        Q[Q < 1e-60] = 1e-15;
        Q[Q > 1 - 1e-60] = 1e-15;

        for k in range(self._NumComponent):
            weight[k] = np.mean(Q[:, k]);

            if self._Distribution == 'normal':
                mu[k]     = np.sum(Q[:, k] * data) / np.sum(Q[:, k]);
                sigma[k]  = np.sqrt( np.sum(Q[:, k] * np.square(data - mu[k])) / np.sum(Q[:, k]) );
            elif self._Distribution == 'lognormal':
                mu[k]     = np.sum(Q[:, k] * np.log(data)) / np.sum(Q[:, k]);
                sigma[k]  = np.sqrt( np.sum(Q[:, k] * np.square(np.log(data) - mu[k])) / np.sum(Q[:, k]) );

    def UpdateGeneration_EM(self, data, generation):
        Fitness = [0] * self._NumSolution;

        for i in range(self._NumSolution):
            # shallow copy
            SingleSolution = generation[i,:];
            Mu = SingleSolution[0:self._NumComponent];
            Sigma = SingleSolution[self._NumComponent : self._NumComponent * 2];
            Weight = SingleSolution[self._NumComponent * 2 : self._NumComponent * 3];
            
            if i in range(self._NumElitism):
                Fitness[i] = -1 * MixtureModel.Loglikelihood(data, self._Distribution, self._NumComponent, Mu, Sigma, Weight);
            else:
                self.PerformEM(data, Mu, Sigma, Weight);
                Fitness[i] = -1 * MixtureModel.Loglikelihood(data, self._Distribution, self._NumComponent, Mu, Sigma, Weight);

        return Fitness; # the variable 'generation' is permanently altered in this function

    def SortGeneration(self, generation, sortIndex):
        """ sort generation with sorted fitness index """
        return generation[sortIndex, :];

    def Elitism(self, sortedGeneration):
        """ the best two solutions will be definitely kept  """
        return sortedGeneration[range(self._NumElitism), :];

    def RankSelection(self, sortedGeneration):
        """ 
        the input is the sorted genreation
        the solution with the best fitness has the greatest likelihood to be selected
        the solution with the worst fitness has the least likelihood to be selected
        this function is to select good solutions based on probability
        the output is an array with several indexes
        """
        Summation = np.sum(range(self._NumSolution));
        SolutionProbability = [0] * self._NumSolution;
        for i in range(self._NumSolution):
            SolutionProbability[i] = ( (self._NumSolution - 1 - i) / Summation );
        
        SelectedIndex = np.random.choice(range(self._NumSolution), size = self._NumElitism, p = SolutionProbability);
        return SelectedIndex;

    def Crossover(self, selectedSolutions):
        """ this function performs uniform crossover """
        RandomNumber = np.random.uniform(0, 1);

        if RandomNumber <= self._CrossoverRate:
            for i in range(0, self._NumComponent * 2):
                # Option = np.random.uniform(0, 1);
                # if Option <= 0.5:
                Temp = selectedSolutions[0, i];
                selectedSolutions[0, i] = selectedSolutions[1, i];
                selectedSolutions[1, i] = Temp;

        return selectedSolutions;
         
    def Mutate(self, crossedSolutions):
        """ this function performs mutation
        the variables in a solution have a chance to be added by a random value from Gaussian(0, 1)
        """
        RandomNumber = np.random.uniform(0, 1);

        if RandomNumber <= self._MutationRate:
            # Option = np.random.uniform(0, 1);
            # if Option <= 0.5:
            crossedSolutions[0:2, 0:self._NumComponent] += np.random.normal(0, 0.05, (2,self._NumComponent));
            crossedSolutions[0:2, self._NumComponent:self._NumComponent*2] += np.random.normal(0, 0.05, (2, self._NumComponent));
            crossedSolutions[crossedSolutions < 0] = RandomNumber;
        return crossedSolutions;

    def SelectedSolutions(self, sortedGeneration):
        """
        after keeping the best two solutions, select two solutions in the current generation using the ranking selection strategy
        """
        SelectedIndexes = self.RankSelection(sortedGeneration);
        return(sortedGeneration[SelectedIndexes, :]);

    def NewGeneration(self, data, sortedGeneration):
        # perform elitism
        Generation = self.Elitism(sortedGeneration);

        for i in range(int(self._NumSolution / 2) - 1):
            SelectedSolutions = self.SelectedSolutions(sortedGeneration);
            CrossedSolutions = self.Crossover(SelectedSolutions);
            MutatedSolutions = self.Mutate(SelectedSolutions);
            Generation = np.append(Generation, MutatedSolutions, axis = 0);
        
        return Generation;

    def Execute(self, data):
        NumIteration = 100;
        TrackBestFitness = [0] * NumIteration;
        Generation = self.InitiallizeGeneration(data);
        Fitness = self.UpdateGeneration_EM(data, Generation);

        for i in range(NumIteration):
            SortedFitness = np.sort(Fitness);
            IndexReturn = np.argsort(Fitness);
            TrackBestFitness[i] = SortedFitness[0];
            SortedGeneration = Generation[IndexReturn, :];
            
            Generation = self.NewGeneration(data, SortedGeneration);
            Fitness = self.UpdateGeneration_EM(data, Generation);

        Mu = Generation[0, 0:self._NumComponent];
        Sigma = Generation[0, self._NumComponent : self._NumComponent * 2];
        Weight = Generation[0, self._NumComponent * 2 : self._NumComponent * 3];
        Loglikelihood = TrackBestFitness;

        return Mu, Sigma, Weight, Loglikelihood;

