#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np 
import time
import math
import random
from scipy import linalg as LA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import statsmodels.api as sm
from termcolor import colored
import similaritymeasures
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean


# In[2]:


def read_file(file_name):
    data = []
    Data = []
    flag = True
    with open(file_name, "r") as f:
        for line in f:
            item = line.strip().split("|")
            if flag == True and len(item) == 5:
                data.append([float(item[0]), float(item[1])])
                Data.append([float(item[3]), float(item[4])])
                flag = False
            elif flag == True and len(item) == 6:
                data.append([float(item[0]), float(item[1])])
                Data.append([float(item[3]), float(item[4])])
                flag = False
            else:
                flag = True
    return np.array(data), np.array(Data)


# In[3]:


P1 = glob.glob('/Users/hasan/Desktop/Anaconda/Research/gpsdata/person1/**/', recursive=True)[1:]
P2 = glob.glob('/Users/hasan/Desktop/Anaconda/Research/gpsdata/person2/**/', recursive=True)[1:]
I1 = glob.glob('/Users/hasan/Desktop/Anaconda/Research/gpsdata/person1/**/*.txt', recursive=True)
I2 = glob.glob('/Users/hasan/Desktop/Anaconda/Research/gpsdata/person2/**/*.txt', recursive=True)


# In[4]:


A = pd.read_csv(I1[0], header=None, delimiter="|")
np.array(A)


# In[5]:


len(P1), len(P2), len(I1), len(I2), len(I1)/len(I2)


# In[6]:


data1_lan_long = [0] * len(I1) # trajectories in lan,long-coordinates
data1_x_y = [0] * len(I1) # trajectories in projected x,y-coordinate
data2_lan_long = [0] * len(I2) # trajectories in lan,long-coordinates
data2_x_y = [0] * len(I2) # trajectories in projected x,y-coordinate

for i in range(len(I1)):
    z =read_file(I1[i])
    data1_lan_long[i] = z[0]
    data1_x_y[i] = z[1]

for i in range(len(I2)):
    z =read_file(I2[i])
    data2_lan_long[i] = z[0]
    data2_x_y[i] = z[1]
    
data1_lan_long = np.array(data1_lan_long, dtype = 'object')
data1_x_y = np.array(data1_x_y, dtype = 'object')
data2_lan_long = np.array(data2_lan_long, dtype = 'object')
data2_x_y = np.array(data2_x_y, dtype = 'object')


# In[7]:


data1_lan_long = data1_x_y
data2_lan_long = data2_x_y


# In[8]:


def remove_segments(traj): # removes stationary points
    p2 = traj[1:]
    p1 = traj[:-1]
    L = ((p2-p1)*(p2-p1)).sum(axis =1)
    I = np.where(L>1e-16)[0]
    return traj[I]


# In[9]:


data1_lan_long = np.array([remove_segments(data1_lan_long[i]) for i in range(len(data1_lan_long))])

data2_lan_long = np.array([remove_segments(data2_lan_long[i]) for i in range(len(data2_lan_long))])

len(data1_lan_long), len(data2_lan_long)


# In[11]:


L1 = np.array([len(data1_lan_long[i]) for i in range(len(data1_lan_long))])
L2 = np.array([len(data2_lan_long[i]) for i in range(len(data2_lan_long))])
    
len(np.where(L1 < 2500)[0]), len(np.where(L2 < 2000)[0])


# In[13]:


np.sort(L1), np.sort(L2)


# In[12]:


np.mean(L1), np.mean(L2)


# In[11]:


def get_mu(data_1, data_2):
    a = np.mean([np.mean(data_1[i], 0) for i in range(len(data_1))], 0)
    b = np.mean([np.mean(data_2[i], 0) for i in range(len(data_2))], 0)
    c = abs(a-b)
    return max(c)


# # KNN with Soft-DTW with matrix saving method

# In[12]:


def calculate_dists_soft_dtw(data1, data2, gamma, path): 
    start_time = time.time() 
    data = np.concatenate((data1, data2), 0)
    n = len(data)
    A = []
    for i in range(n-1):
        for j in range(i+1, n):
            D = SquaredEuclidean(data[i], data[j])
            sdtw = SoftDTW(D, gamma=gamma)
            A.append(sdtw.compute())
    A = np.array(A)
    tri = np.zeros((n, n))
    tri[np.triu_indices(n, 1)] = A
    for i in range(1, n):
        for j in range(i):
            tri[i][j] = tri[j][i]
    np.savetxt(path, tri, delimiter=',')
    total_time = time.time() - start_time
    return total_time


# In[19]:


path = '/Users/hasan/Desktop/Anaconda/Research/Calculated Distance Matrices for KNN/persons-soft-dtw.csv'
calculate_dists_soft_dtw(data1_lan_long, data2_lan_long, gamma=1e-10, path=path)


# In[12]:


def KNN_with_dists_soft_dtw(n_1, n_2, path_to_dists):
    '''path example: '/content/gdrive/My Drive/traj-dist/Calculated Distance Matrices (car-bus)/sspd.csv'
       path_to_dists: the path to the corresponding distance matrix
       n_1: len(data_1)
       n_2: len(data_2)'''

    I_1, J_1, y_train_1, y_test_1 = train_test_split(np.arange(n_1), 
                                                np.ones(n_1), test_size=0.3)
    I_2, J_2, y_train_2, y_test_2 = train_test_split(np.arange(n_1, n_1+n_2), 
                                                np.ones(n_2), test_size=0.3)
    labels = np.array([1] * n_1 + [0] * n_2)
    I = np.concatenate((I_1, I_2), 0)
    np.random.shuffle(I)
    J = np.concatenate((J_1, J_2), 0)
    np.random.shuffle(J)

    dist_matrix = np.array(pd.read_csv(path_to_dists,  header=None))

    D_train = dist_matrix[I][:, I]
    D_test = dist_matrix[J][:,I]
    train_labels = labels[I]
    test_labels = labels[J]

    clf = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
    
    #Train the model using the training sets
    clf.fit(D_train, list(train_labels))

    #Predict labels for train dataset
    train_pred = clf.predict(D_train)
    train_error = sum(train_labels != train_pred)/len(I)
    
    #Predict labels for test dataset
    test_pred = clf.predict(D_test)
    test_error = sum((test_labels != test_pred))/len(J)
        
    return train_error, test_error


# In[13]:


def KNN_average_error_soft_dtw(data1, data2, num_trials, path_to_dists):

    '''path_to_dists: the path to the corresponding distance matrix'''

    Start_time = time.time()

    train_errors = np.zeros(num_trials)
    test_errors = np.zeros(num_trials)

    for i in range(num_trials):
        train_errors[i], test_errors[i] = KNN_with_dists_soft_dtw(len(data1), len(data2), path_to_dists)

    Dict = {}
    Dict[1] = [f"KNN with soft dtw", 
                    np.round(np.mean(train_errors), decimals = 4), 
                    np.round(np.mean(test_errors), decimals = 4), 
                    np.round(np.std(test_errors), decimals = 4)]

    df = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier',
                                'Train Error', 'Test Error', 'std'])
    print(colored(f"num_trials = {num_trials}", "blue"))
    print(colored(f'total time = {time.time() - Start_time}', 'green'))

    return (df, np.mean(train_errors), np.mean(test_errors), np.std(test_errors))


# In[18]:


path = '/Users/hasan/Desktop/Anaconda/Research/Calculated Distance Matrices for KNN/persons-soft-dtw.csv'
A = KNN_average_error_soft_dtw(data1_lan_long, data2_lan_long, num_trials=50, 
                               path_to_dists=path)
A[0]


# # Runtime Analysis

# In[13]:


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

#metrics = ['discret_frechet', 'hausdorff', 'dtw', SoftDTW, 'sspd', 'erp', 'edr', 'lcss',  
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
        if str(self.metric) == str(SoftDTW): 
            A = []
            for i in range(n-1):
                for j in range(i+1, n):
                    D = SquaredEuclidean(data[i], data[j])
                    sdtw = self.metric(D, gamma=self.gamma)
                    A.append(sdtw.compute())
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
    


# In[15]:


metrics = ['discret_frechet', 'hausdorff', dtw_tslearn, fastdtw, 'lcss', 'sspd',
           'edr', 'erp', 'd_Q_pi']
metrics = [SoftDTW]
runtimes = []
for metric in metrics:
    get_ipython().run_line_magic('timeit', 'KNN_runTime(data1_lan_long, data2_lan_long, metric=metric, gamma=1e-10,                         eps_edr=0.1, eps_lcss=0.1, Q_size=20, Q=None, p=2, n_neighbors=5,                         pair=[0,1])')

    Runtime = KNN_runTime(data1_lan_long, data2_lan_long, metric=metric, gamma=1e-10, 
                          eps_edr=0.02, eps_lcss=0.02, Q_size=20, Q=None, p=2, 
                          n_neighbors=5, pair=[0,1])
    a = Runtime.KNN_runtime()
    runtimes.append(a)
    print(colored(f'runtimes for {metric}: {a}', 'magenta'))
    print(colored("===============================================================", 'red'))

print(colored(runtimes, 'blue'))


# In[ ]:




