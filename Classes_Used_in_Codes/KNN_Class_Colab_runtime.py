'''This class handles all the metrics in "metrics array bellow" and is appropriate for using in Anaconda 
   for example, but not on Google Colab.'''

'''Requirements: (These are already installed in my computer)
        1. We need "pip install trjtrypy" in order to be able to use d_Q_pi
        2. We need "pip install tslearn" in order to be able to use dtw
        3. We need "pip install fastdtw" in order to be able to use fastdtw
        4. We need "pip install traj_dist" in order to be able to use the rest of metrics'''


import numpy as np
import time
import pandas as pd
import random
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import traj_dist.distance as tdist
import pickle
import tslearn
from tslearn.metrics import dtw as dtw_tslearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from trjtrypy.distances import d_Q_pi
from termcolor import colored

#metrics = ['discret_frechet', 'hausdorff', 'dtw', 'sspd', 'erp', 'edr', 'lcss',  
#           fastdtw, dtw, d_Q_pi]

# path example: 
#'Calculated Distance Matrices for KNN/Beijing-Pairs['+str(pairs_final[i])+']-d_Q_pi.csv'


class KNN_runTime:
    
    def __init__(self, data1, data2, metric, gamma=None, eps_edr=None, eps_lcss=None, 
                 Q_size=None, Q=None, p=2, n_neighbors=5, pair=None):
        '''data1 = data[pair[0]]
           data2 = data[pair[1]]'''
        self.data1 = data1
        self.data2 = data2
        self.metric = metric
        self.gamma = gamma
        self.eps_edr = eps_edr
        self.eps_lcss = eps_lcss
        self.Q_size = Q_size
        self.Q = Q
        self.p = p
        self.n_neighbors = n_neighbors
        self.pair = pair
    



    def calculate_dists_matrix(self):

        data = np.concatenate((self.data1, self.data2), 0)
        n = len(data)

        if self.metric == 'lcss':
            A = tdist.pdist(data, self.metric, type_d="euclidean", eps=self.eps_lcss)
        if self.metric == 'edr':
            A = tdist.pdist(data, self.metric, type_d="euclidean", eps=self.eps_edr)
        if self.metric in ['discret_frechet', 'hausdorff', 'dtw', 'sspd', 'erp']: 
            A = tdist.pdist(data, str(self.metric))
        if self.metric == fastdtw: 
            A = []
            for i in range(n-1):
                for j in range(i+1, n):
                    A.append(self.metric(data[i], data[j])[0])
        if self.metric == dtw_tslearn: 
            A = []
            for i in range(n-1):
                for j in range(i+1, n):
                    A.append(self.metric(data[i], data[j]))
        if self.metric == 'd_Q_pi':
            A = []
            if self.Q_size:
                Q = self.generate_random_Q()
                for i in range(n-1):
                    for j in range(i+1, n):
                        A.append(d_Q_pi(Q, data[i], data[j], p=self.p))
            elif len(self.Q):
                for i in range(n-1):
                    for j in range(i+1, n):
                        A.append(d_Q_pi(self.Q, data[i], data[j], p=self.p))

        tri = np.zeros((n, n))
        tri[np.triu_indices(n, 1)] = np.array(A)
        for i in range(1, n):
            for j in range(i):
                tri[i][j] = tri[j][i]
                
        return tri





    '''The following function is only used for d_Q_pi distance in order to 
       generate random landmarks
       Notice: in this pattern of coding we cannot use train1 and train2 to 
       get Q in the following function.'''
    def generate_random_Q(self):
        Q = np.zeros((self.Q_size, 2))
        data = np.concatenate((self.data1, self.data2), 0)
        Mean = np.mean([np.mean(data[i], 0) for i in range(len(data))], 0)
        Std = np.std([np.std(data[i], 0) for i in range(len(data))], 0)
        Q[:,0] = np.random.normal(Mean[0], 4 * Std[0], self.Q_size)
        Q[:,1] = np.random.normal(Mean[1], 4 * Std[1], self.Q_size)
        return Q




    def KNN_runtime(self):
        
        start_time = time.time()
        n_1 = len(self.data1)
        n_2 = len(self.data2) 
        
        labels = np.array([1] * n_1 + [-1] * n_2)
        data = np.concatenate((self.data1, self.data2), 0)
        dist_matrix = self.calculate_dists_matrix()

        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric='precomputed')
        clf.fit(dist_matrix, list(labels))
        stop_time = time.time()
        runtime = stop_time - start_time

        return runtime
    