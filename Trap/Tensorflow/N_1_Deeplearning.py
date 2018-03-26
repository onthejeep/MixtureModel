import tensorflow as tf;
import numpy as np;
import os;
import PenaltyImpe as PenaltyImpe;
import SaveData as SaveData;
import sklearn.cluster as cl;
import N_1_MixtureModel;

class MLE:

    def __init__(self):
        self._LearningRate_Decay = np.repeat(0.001, 500);
        self._LearningRate_Decay = np.append(self._LearningRate_Decay, np.repeat(0.0005, 300));
        self._LearningRate_Decay = np.append(self._LearningRate_Decay, np.repeat(0.0002, 100));
        self._TrainStep = len(self._LearningRate_Decay);
        self._Distribution = 'normal';
        
        self._NumComponent = 2;
        self._Mu        = [None] * self._NumComponent;
        self._Sigma_Phi = [None] * self._NumComponent;    
        self._Sigma     = [None] * self._NumComponent;
        self._W_Phi     = [None] * self._NumComponent;    
        self._W         = [None] * self._NumComponent;
        self._Data_Placeholder = tf.placeholder(dtype = tf.float32);
        self._LearningRate_Placeholder = tf.placeholder(tf.float32, shape = ());
        self._Hypothesis = None;
        self._Cost = None;
        self._Optimizer = None;
        self._DataSize_Placeholder = tf.placeholder(tf.float32, shape = ());

    def DefineFlow(self):
        for i in range(self._NumComponent):
            self._Mu[i] = tf.Variable(1.0, trainable = True, dtype = tf.float32, name = 'Mu{}'.format(i));
            self._Sigma_Phi[i] = tf.Variable(0.5, trainable = True, dtype = tf.float32, name = 'Sigma_Phi{}'.format(i));
            self._Sigma[i] = tf.square(self._Sigma_Phi[i], name = 'Sigma{}'.format(i));
            self._W_Phi[i] = tf.Variable(0.5, trainable = True, dtype = tf.float32, name = 'W_Phi{}'.format(i));
            self._W[i] = tf.square(self._W_Phi[i], name = 'W{}'.format(i));

        self._Hypothesis = self.Loglikelihood(self._Data_Placeholder,
            mu = self._Mu, sigma = self._Sigma, weight = self._W);

        self._Cost = -1.0 * tf.reduce_sum(self._Hypothesis) / self._DataSize_Placeholder;
        self._Cost += PenaltyImpe.Penalty.PenaltyFunction(weights = self._W, iterationIndex = 1);

        self._Optimizer = tf.train.AdamOptimizer(self._LearningRate_Placeholder).minimize(self._Cost);


    def NormalPdf(self, data, mu, sigma):
        Exponential = -1 * tf.square(data - mu) / (2 * tf.square(sigma));
        Probability = 1 / (tf.sqrt(2 * np.pi * tf.square(sigma))) * tf.exp(Exponential);
        return Probability;

    def LognormalPdf(self, data, mu, sigma):
        Exponential = -1 * tf.square(tf.log(data) - mu) / (2 * tf.square(sigma));
        Probability = 1 / (tf.sqrt(2 * np.pi) * sigma * data) * tf.exp(Exponential);
        return Probability;

    def Loglikelihood(self, data, mu, sigma, weight):
        Result = 0;
        for i in range(self._NumComponent - 1):
            Result += weight[i] * self.NormalPdf(data, mu[i], sigma[i]);
        Result += weight[-1] * self.LognormalPdf(data, mu[-1], sigma[-1]);

        return tf.log(tf.clip_by_value(Result, 1e-10, 100));

    def GetTrainableVariable(self, tfSess):
        VariablesNames = [v.name for v in tf.trainable_variables()];
        Values = tfSess.run(VariablesNames);
        for name, value in zip(VariablesNames, Values):
            print('Variable: ', name);

    def Training(self, data):
        # tf.summary.FileWriterCache.clear();

        InitialMu, InitialStd, InitialWeight = N_1_MixtureModel.InitializeVariable(data, self._NumComponent);
        Cost_Eval = [];
        
        # with tf.name_scope('Summary'):
        #     Hist_W = [None] * self._NumComponent;
        #     Hist_Mu = [None] * self._NumComponent;
        #     Hist_Sigma = [None] * self._NumComponent;

        #     for i in range(self._NumComponent):
        #         Hist_W[i] = tf.summary.scalar('weights{}'.format(i), self._W[i]);
        #         Hist_Mu[i] = tf.summary.scalar('mu{}'.format(i), self._Mu[i]);
        #         Hist_Sigma[i] = tf.summary.scalar('sigma{}'.format(i), self._Sigma[i]);
        
        # with tf.name_scope('Cost-Function'):
        #     Scalar_Cost = tf.summary.scalar('cost', self._Cost);

        with tf.Session() as Sess:
            Sess.run(tf.global_variables_initializer());

            # SummaryWriter = tf.summary.FileWriter('C:/Users/onthejeep/graph', Sess.graph);  # create writer
            # Merged_summary_op = tf.summary.merge_all();

            for i in range(self._NumComponent):
                Sess.run(tf.assign(self._Mu[i], InitialMu[i]));
                Sess.run(tf.assign(self._Sigma_Phi[i], np.sqrt(InitialStd[i])));
                Sess.run(tf.assign(self._W_Phi[i], np.sqrt(InitialWeight[i])));

            for i in range(self._TrainStep):
                C, _, Mu_Eval, Sigma_Eval, W_Eval = \
                        Sess.run([self._Cost, self._Optimizer, self._Mu, 
                        self._Sigma, self._W],
                        feed_dict = {self._Data_Placeholder: data,
                                    self._LearningRate_Placeholder: self._LearningRate_Decay[i],
                                    self._DataSize_Placeholder: len(data)
                                    });
                Cost_Eval.append(C);
                # SummaryWriter.add_summary(Summary, i);

            # SummaryWriter.close();

        return Mu_Eval, Sigma_Eval, W_Eval, Cost_Eval;


# tensorboard --logdir = D:\MySVN\UA-Research\EM_Gradient_PSO\Analysis\MLE_GradientDescent