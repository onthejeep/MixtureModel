'''save data (Traveltime)'''

import tensorflow as tf;
import numpy as np;

tf.app.flags.DEFINE_float('mu1', 4, 'mode1-mu');
tf.app.flags.DEFINE_float('mu2', 7, 'mode2-mu');
tf.app.flags.DEFINE_float('mu3', 10, 'mode3-mu');

tf.app.flags.DEFINE_float('std1', 1, 'mode1-std');
tf.app.flags.DEFINE_float('std2', 2, 'mode2-std');
tf.app.flags.DEFINE_float('std3', 3, 'mode3-std');

tf.app.flags.DEFINE_float('w1', 0.6, 'mode1-weight');
tf.app.flags.DEFINE_float('w2', 0.3, 'mode2-weight');
tf.app.flags.DEFINE_float('w3', 0.1, 'mode3-weight');

tf.app.flags.DEFINE_integer('datasize', 100, 'data-size');
tf.app.flags.DEFINE_integer('number_mixture', 3, 'number of mixture component');
Flags = tf.app.flags.FLAGS;

def FakeData():
    '''simulated data'''
    Traveltime = np.random.normal(Flags.mu1, Flags.std1, size = int(Flags.datasize * Flags.w1));
    Traveltime = np.append(Traveltime, np.random.normal(Flags.mu2, Flags.std2, size = int(Flags.datasize * Flags.w2)));
    Traveltime = np.append(Traveltime, np.random.normal(Flags.mu3, Flags.std3, size = int(Flags.datasize * Flags.w3)));
    Traveltime = Traveltime[Traveltime > 0];
    return Traveltime;


# ground truth statistics
# mean = 0.6 * 4 + 0.3 * 7 + 0.1 * 10 = 5.5;
# mu_i - mu = [-1.5, 1.5, 4.5];
# variance = 0.6 * (1.5^2 + 1^2) + 0.3 * (1.5^2 + 2^2) + 0.1 * (1.5^2 + 3^2) = 4.95
# skewness = 0.6 * (-1.5^3 - 3*1.5*1^2) + 0.3 * (1.5^3 + 3 * 1.5 * 2^2) + 0.1 * (4.5^3 + 3*4.5*3^2) = 22.95