'''This class does not handle soft-dtw as it is designed to be used in Google Colab. Note that soft-dtw 
   package was not installable on Google Colab.'''

'''Requirements:
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
#           fastdtw, 'dtw_tslearn', 'd_Q_pi']

# path example: 
#'Calculated Distance Matrices for KNN/Beijing-Pairs['+str(pairs_final[i])+']-d_Q_pi.csv'



'''Note: In d_Q_pi distance, if Q_size is given, then Q will not be considered. Otherwise, Q should be given.'''


class KNN:
    def __init__(self, data1, data2, metric, gamma=None, eps_edr=None, eps_lcss=None, 
                 Q_size=None, Q=None, p=2, path=None, test_size=0.3, n_neighbors=5, 
                 num_trials=1000, pair=None):
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
        self.path = path
        self.test_size = test_size
        self.n_neighbors = n_neighbors
        self.num_trials = num_trials
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
    




    def write_matrix_to_csv(self):

        matrix = self.calculate_dists_matrix()
        np.savetxt(self.path, matrix, delimiter=',')
        
        return matrix.shape





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




    
    '''train-test split
       Note: I know I'm using [1] for both labels. Don't worry, it is correct.'''

    def train_test(self):
        
        n_1 = len(self.data1)
        n_2 = len(self.data2) 
        train_idx_1, test_idx_1, train_label_1, test_label_1 = \
                train_test_split(np.arange(n_1), np.ones(n_1), test_size=self.test_size) 
        train_idx_2, test_idx_2, train_label_2, test_label_2 = \
        train_test_split(np.arange(n_1,n_1+n_2), np.ones(n_2), test_size=self.test_size)
        
        return train_idx_1, test_idx_1, train_label_1, test_label_1, \
               train_idx_2, test_idx_2, train_label_2, test_label_2





    def KNN_Classifier(self):
        
        n_1 = len(self.data1)
        n_2 = len(self.data2) 
        train_idx_1, test_idx_1, train_label_1, test_label_1, \
            train_idx_2, test_idx_2, train_label_2, test_label_2 = self.train_test()
        
        labels = np.array([1] * n_1 + [-1] * n_2)
        train_idx = np.concatenate((train_idx_1, train_idx_2), 0)
        np.random.shuffle(train_idx)
        test_idx = np.concatenate((test_idx_1, test_idx_2), 0)
        np.random.shuffle(test_idx)
        
        if self.path:
            dist_matrix = np.array(pd.read_csv(self.path, header=None))
        else:
            print("Warning: There is no pre-calculated distance matrix")
            dist_matrix = self.calculate_dists_matrix()
        
        D_train = dist_matrix[train_idx][:, train_idx]
        D_test = dist_matrix[test_idx][:, train_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric='precomputed')

        #Train the model using the training sets
        clf.fit(D_train, list(train_labels))

        #Predict labels for train dataset
        train_pred = clf.predict(D_train)
        train_error = sum(train_labels != train_pred)/len(train_idx)

        #Predict labels for test dataset
        test_pred = clf.predict(D_test)
        test_error = sum((test_labels != test_pred))/len(test_idx)

        return dist_matrix, train_error, test_error
    
    
    
    def KNN_average_error(self):

        Start_time = time.time()
        train_errors = np.zeros(self.num_trials)
        test_errors = np.zeros(self.num_trials)
        
        for i in range(self.num_trials):
            dist_matrix, train_errors[i], test_errors[i] = self.KNN_Classifier()

        Dict = {}
        Dict[1] = [f"KNN; {self.metric}; pair {self.pair}", 
                        np.round(np.mean(train_errors), decimals=4), 
                        np.round(np.mean(test_errors), decimals=4),
                        np.round(np.median(test_errors), decimals=4),
                        np.round(np.std(test_errors), decimals=4)]

        df = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier',
                            'Train Error', 'Mean', 'Median', 'Std'])
        print(colored(f"num_trials = {self.num_trials}", "blue"))
        print(colored(f'total time = {time.time() - Start_time}', 'green'))

        return (df, np.mean(train_errors), np.mean(test_errors), np.median(test_errors), 
                np.std(test_errors))