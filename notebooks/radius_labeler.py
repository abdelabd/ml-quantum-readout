# See: https://arxiv.org/pdf/2212.03895, Algorithm 1

import numpy as np
import json
import copy

class RadiusLabeler(object):
    def __init__(self):
        self.I0_mean = None
        self.Q0_mean = None
        self.I1_mean = None
        self.Q1_mean = None

    def fit(self, X, y):
        idx0 = np.where(y==0)[0]
        idx1 = np.where(y==1)[0]

        X0 = X[idx0]
        X1 = X[idx1]

        X0_I_tmean, X0_Q_tmean = self.split_and_time_average(X0)
        self.X0_centroid = (np.mean(X0_I_tmean), np.mean(X0_Q_tmean))

        X1_I_tmean, X1_Q_tmean = self.split_and_time_average(X1)
        self.X1_centroid = (np.mean(X1_I_tmean), np.mean(X1_Q_tmean))

        self.r2_0_1 = (self.X0_centroid[0] - self.X1_centroid[0])**2 + (self.X0_centroid[1] - self.X1_centroid[1])**2
        
    def label(self, X, y):
        idx0 = np.where(y==0)[0]
        idx1 = np.where(y==1)[0]

        idx_excite = []
        for idx in idx0:
            X_i = X[idx]
            r2_1_i = (self.X1_centroid[0] - X_i[0])**2 + (self.X1_centroid[1] - X_i[1])**2
            if r2_1_i <= self.r2_0_1: 
                idx_excite.append(idx)

        idx_relax = []
        for idx in idx1:
            X_i = X[idx]
            r2_0_i = (self.X0_centroid[0] - X_i[0])**2 + (self.X0_centroid[1] - X_i[1])**2
            if r2_0_i <= self.r2_0_1:
                idx_relax.append(idx)

        y_out = copy.deepcopy(y)
        y_out[idx_excite] = 2
        y_out[idx_relax] = 3
        print(f"0 = |0>, 1 = |1>, 2 = excitation-error, 3 = relaxation-error")
        return y_out
                
        
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
