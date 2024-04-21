# See: https://github.com/openquantumhardware/qick/blob/main/qick_demos/06_qubit_demos.ipynb

import numpy as np
import json

class ThresholdModel(object):
    def __init__(self):
        self.theta = None
        self.threshold = None

    def fit(self, X, y, nbins=200, ran=1000):

        # Split into ground, excited samples 
        idx0 = np.where(y==0)[0]
        idx1 = np.where(y==1)[0]

        X0 = X[idx0, :]
        X1 = X[idx1, :]

        # Split into I and Q timeseries, average over time-axis
        X0_I, X0_Q = self.split_and_time_average(X0)
        X1_I, X1_Q = self.split_and_time_average(X1)

        # Compute theta
        X0_I_median = np.median(X0_I)
        X0_Q_median = np.median(X0_Q)
        X1_I_median = np.median(X1_I)
        X1_Q_median = np.median(X1_Q)

        ###################################################################################
        self.theta = -np.arctan2((X1_Q_median - X0_Q_median), (X1_I_median - X0_I_median))
        ###################################################################################
        
        # Rotate IQ clusters by theta
        X0_I_prime = X0_I*np.cos(self.theta) - X0_Q*np.sin(self.theta)
        X0_Q_prime = X0_I*np.sin(self.theta) + X0_Q*np.cos(self.theta)
        X1_I_prime = X1_I*np.cos(self.theta) - X1_Q*np.sin(self.theta)
        X1_Q_prime = X1_I*np.sin(self.theta) + X1_Q*np.cos(self.theta)

        # Compute new means of IQ clusters
        X0_I_prime_median = np.median(X0_I_prime)
        X0_Q_prime_median = np.median(X0_Q_prime)
        X1_I_prime_median = np.median(X1_I_prime)
        X1_Q_prime_median = np.median(X1_Q_prime)
                                
        # Compute threshold
        xlims = [X0_I_prime_median - ran, X0_I_prime_median + ran]
        ylims = [X0_Q_prime_median - ran, X0_Q_prime_median + ran]
    
        hist_0, bin_edges_0 = np.histogram(X0_I_prime, bins = nbins, range = xlims)
        hist_1, bin_edges_1 = np.histogram(X1_I_prime, bins = nbins, range = xlims)

        contrast = np.abs(((np.cumsum(hist_0) - np.cumsum(hist_1)) / (0.5*hist_0.sum() + 0.5*hist_1.sum())))
        tind = contrast.argmax()

        ###################################
        self.threshold = bin_edges_0[tind]
        ###################################
    
    def predict(self, X):
        if self.theta == None:
            raise NotImplementedError("Model has not been trained yet")

        # Split into I, Q
        X_I, X_Q = self.split_and_time_average(X)

        # Rotate by theta 
        X_I_prime = X_I*np.cos(self.theta) - X_Q*np.sin(self.theta)

        # Discriminate based on threshold
        qubit_states = np.array(X_I_prime > self.threshold, dtype=int)
        return qubit_states

    def save(self, fpath):
        assert '.json' in fpath
        save_dict = {"theta": self.theta, "threshold": self.threshold}
        with open(fpath, "w") as f:
            json.dump(save_dict, f)

    def load(self, fpath):
        assert '.json' in fpath
        with open(fpath, 'r') as f:
            load_dict = json.load(f)
        self.theta = load_dict['theta']
        self.threshold = load_dict['threshold']
            

    def split_and_time_average(self, X):
        # X: [N_SAMPLES, TIMESERIES_LENGTH]

        # Split into I, Q
        I_idx = [i for i in range(X.shape[1]) if i%2==0]
        Q_idx = [i for i in range(X.shape[1]) if i%2==1]
    
        X_I = X[:, I_idx]
        X_Q = X[:, Q_idx]

        # Average over time-axis
        X_I_mean = np.mean(X_I, axis=1)
        X_Q_mean = np.mean(X_Q, axis=1)
    
        return X_I_mean, X_Q_mean
