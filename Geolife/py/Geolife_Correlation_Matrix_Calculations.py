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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
import statsmodels.api as sm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from termcolor import colored
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.svm import LinearSVC
from datetime import datetime
from scipy.stats import entropy
import os
import pickle
import trjtrypy as tt
from trjtrypy.distances import d_Q
from trjtrypy.distances import d_Q_pi
import trjtrypy.visualizations as vs
from scipy.spatial import distance
from trjtrypy.featureMappings import curve2vec


# In[2]:


def ExpCurve2Vec(points,curves,mu):
    D = tt.distsbase.DistsBase()
    a = np.array([np.exp(-1*np.power(D.APntSetDistACrv(points,curve),2)/(mu)**2) for curve in curves])
    return a


# In[3]:


I = glob.glob('Geolife Trajectories 1.3/**/*.plt', recursive=True)
S = list(set([I[i][:45] for i in range(len(I))]))
user_idx = np.array([S[i][30:33] for i in range(len(S))])
print(user_idx[:10])
len(S), S[0]


# In[4]:


def read_file(path):
    J = glob.glob(path, recursive=True)
    data = [0] * len(J)
    c = 0
    for j in range(len(J)):
        data[j] = []
        with open(J[j], "r") as f:
            for line in f:
                if c > 6:
                    item = line.strip().split(",")
                    if len(item) == 7:
                        data[j].append(np.array([float(item[0]), float(item[1]), 
                                                 float(item[4])]))
                c += 1
        data[j] = np.array(data[j])
    return np.array(data)


# ### data1[i] in the following is user_i from Beijing dataset 
# 
# trajectories have time dimension in data1

# In[5]:


start_time = time.time()
user_idx_int = np.array(list(map(int, user_idx)))
data1 = [0] * len(S)
for i in range(len(S)):
    idx = np.where(user_idx_int == i)[0][0]
    path = S[idx]+'*.plt'
    J = glob.glob(path, recursive=True)
    data1[i] = read_file(path)
data1 = np.array(data1)
print(time.time() - start_time)


# In[6]:


data_1 = data1 + 0
len(data_1)


# In[7]:


def remove_segments(traj): # removes stationary points
    p2 = traj[:,:2][1:]
    p1 = traj[:,:2][:-1]
    L = ((p2-p1)*(p2-p1)).sum(axis =1)
    I = np.where(L>1e-16)[0]
    return traj[I]


# ### In data_1 below:
#     (1) there is no stationary points
#     (2) there is no trajectories with more than 200 waypoints

# In[8]:


for i in range(len(data_1)):
    data_1[i] = np.array(list(map(remove_segments, data_1[i])), dtype='object')
    L = np.array([len(data_1[i][j]) for j in range(len(data_1[i]))])
    I = np.where((L > 1000))[0]
    data_1[i] = data_1[i][I]

I = np.where(np.array([len(data_1[i]) for i in range(len(data_1))]) > 0)[0]
data_1 = data_1[I]
print("len(data_1) =", len(data_1))
print("selected users: \n", I)


# # Partitioning trajectories to less than 20 minutes long

# In[9]:


def partition(trajectory, threshold=20):
    '''threshold is in minutes'''
    trajectories = []
    a = 24 * 60 * sum(trajectory[:,2][1:] - trajectory[:,2][:-1])
    if a <= threshold:
        return np.array(trajectory.reshape(1, len(trajectory), 3))
    else: 
        i = 0
        while a > threshold:
            j = i + 0
            val = 0
            while val < threshold: 
                if i < len(trajectory) - 1:
                    temp = val + 0
                    val += 24 * 60 * (trajectory[:,2][1:][i] - trajectory[:,2][:-1][i])
                    i += 1
                else: 
                    break
            if len(trajectory[j:i-1]) > 0:
                trajectories.append(trajectory[j:i-1])
            a = a - val
        if len(trajectory[i:]) > 0:
            trajectories.append(trajectory[i:])
    return np.array(trajectories)


# ### data3 below is the array of trajectories having less than 20 minutes long

# In[13]:


data3 = [0] * len(data_1)

for i in range(len(data_1)):
    data3[i] = []
    for j in range(len(data_1[i])):
        A = partition(data_1[i][j], threshold=20)
        for k in range(len(A)):
            data3[i].append(A[k])
    data3[i] = np.array(data3[i], dtype='object')
    
data3 = np.array(data3, dtype='object')

data3.shape, data3[0].shape, data3[0][0].shape


# In[15]:


data4 = data3 + 0


# data4 is the users having between 100 and 200 trajectories and each has length between 10 and 200 trajectory

# In[16]:


for i in range(len(data4)):
    A = np.array([len(data4[i][j]) for j in range(len(data4[i]))])
    I = np.where((A > 10) & (A < 200))[0]
    data4[i] = data4[i][I]
    
print(len(data4))
A = np.array([len(data4[i]) for i in range(len(data4))])
chosen_users = np.where((A > 100) & (A < 200))[0]
data4 = data4[chosen_users]

print("chosen users:", chosen_users)
print("len(data4) =", len(data4))
A = [len(data4[i]) for i in range(len(data4))]
print("length of preprocessed users in data4: \n", np.sort(A))


# ### data2 is the same as data4 but without time dimension

# In[17]:


data2 = data4 + 0
for i in range(len(data2)):
    data2[i] = np.array([data4[i][j][:,:2] for j in range(len(data4[i]))], dtype='object')
len(data2)


# # Classifiers

# In[22]:


CC = [100, 100, 10]
number_estimators = [50, 50]


clf0 = [make_pipeline(LinearSVC(dual=False, C=CC[0], tol=1e-5, 
                               class_weight ='balanced', max_iter=1000)), 
        "SVM, LinearSVC, C = "+str(CC[0])]
clf1 = [make_pipeline(StandardScaler(), svm.SVC(C=CC[1], kernel='rbf', gamma='auto', max_iter=200000)),
        "Gaussian SVM, C="+str(CC[1])+", gamma=auto"]
clf2 = [make_pipeline(StandardScaler(), svm.SVC(C=CC[2], kernel='poly', degree=3, max_iter=400000)),
        "Poly kernel SVM, C="+str(CC[2])+", deg=auto"]
clf3 = [DecisionTreeClassifier(), "Decision Tree"]
clf4 = [RandomForestClassifier(n_estimators=number_estimators[0]), 
         "RandomForestClassifier, n="+str(number_estimators[0])]
clf5 = [KNeighborsClassifier(n_neighbors=5), "KNN"]
clf6 = [LogisticRegression(solver='newton-cg'), "Logistic Regression"]
clf7 = [Perceptron(tol=1e-5, random_state=0, validation_fraction=0.01, 
                               class_weight= "balanced"), "Perceptron"]

clf = [clf0, clf1, clf2, clf3, clf4, clf5, clf6, clf7]
classifs = [item[0] for item in clf]
keys = [item[1] for item in clf]


# # Chosen pairs of users

# In[30]:


pairs_final
#pairs_final = [[4, 12], [4, 16], [5, 12], [8, 10]]


# # Calculations needed for correlation matrix

# ## Classification of pairs_100 with $v_Q$ with random $Q$ in each iteration

# In[64]:


v_Q_errors_100 = []
v_Q_stds_100 = []

for pair in pairs_100:
    print(colored(f"pair={pair}", 'green'))
    classif = binaryClassificationAverageMajority(data[pair[0]], data[pair[1]], Q_size=20,
                                            epoch=100, num_trials_maj=1, classifiers=clf, 
                                            version='unsigned', test_size=0.3)
    A = classif.classification_v_Q()
    print(A[0])
    v_Q_errors_100.append(np.round(A[2], decimals=4).tolist())
    v_Q_stds_100.append(np.round(A[3], decimals=4).tolist())
    print("===========================================================================")

print('v_Q_errors_100 = \n', v_Q_errors_100)
print('v_Q_stds_100 = \n', v_Q_stds_100)
print("===========================================================================")
print(colored(f'average errors: {list(np.round(np.mean(v_Q_errors_100, 0), decimals=4))}', 'blue'))
print(colored(f'average stds: {list(np.round(np.mean(v_Q_stds_100, 0), decimals=4))}', 'green'))

np.savetxt('Claculated test errors for correlation/Beijing/random_V_Q_errors.csv', v_Q_errors_100, 
           delimiter=',')
np.savetxt('Claculated test errors for correlation/Beijing/random_V_Q_stds.csv', v_Q_stds_100, 
           delimiter=',')


# ## Classification for pairs_100 with $v_Q^{\varsigma}$ with random $Q$ in each iteration

# In[147]:


v_Q_sigma_errors_100 = []
v_Q_sigma_stds_100 = []

for pair in pairs_100:
    print(colored(f"pair={pair}", 'green'))
    classif = binaryClassificationAverageMajority(data[pair[0]], data[pair[1]], Q_size=20,
                                            epoch=100, num_trials_maj=1, classifiers=clf, 
                                            version='signed', sigma=10, test_size=0.3)
    A = classif.classification_v_Q()
    print(A[0])
    v_Q_sigma_errors_100.append(np.round(A[2], decimals=4).tolist())
    v_Q_sigma_stds_100.append(np.round(A[3], decimals=4).tolist())
    print("===========================================================================")

print('v_Q_sigma_errors_100 = \n', v_Q_sigma_errors_100)
print('v_Q_sigma_stds_100 = \n', v_Q_sigma_stds_100)
print("===========================================================================")
print(colored(f'average errors: {list(np.round(np.mean(v_Q_sigma_errors_100, 0), decimals=4))}', 'blue'))
print(colored(f'average stds: {list(np.round(np.mean(v_Q_sigma_stds_100, 0), decimals=4))}', 'green'))

np.savetxt('Claculated test errors for correlation/Beijing/random_V_Q_sigma_errors.csv', 
           v_Q_sigma_errors_100, delimiter=',')
np.savetxt('Claculated test errors for correlation/Beijing/random_V_Q_sigma_stds.csv', 
           v_Q_sigma_stds_100, delimiter=',')


# ## Classification for pairs_100 with endpoints

# In[149]:


endpoints_errors_100 = []
endpoints_stds_100 = []
i = 0

for pair in pairs_100:
    print("i=", i)
    print(colored(f"pair={pair}", 'green'))
    classif = binaryClassificationAverageMajority(data[pair[0]], data[pair[1]], Q_size=20,
                                            epoch=100, num_trials_maj=1, classifiers=clf, 
                                            version='unsigned', test_size=0.3)
    A = classif.endpoint_classification()
    print(A[0])
    endpoints_errors_100.append(np.round(A[2], decimals=4).tolist())
    endpoints_stds_100.append(np.round(A[3], decimals=4).tolist())
    print("===========================================================================")
    i += 1

print('endpoints_errors_100 = \n', endpoints_errors_100)
print('endpoints_stds_100 = \n', endpoints_stds_100)
print("===========================================================================")
print(colored(f'average errors: {list(np.round(np.mean(endpoints_errors_100, 0), decimals=4))}', 'blue'))
print(colored(f'average stds: {list(np.round(np.mean(endpoints_stds_100, 0), decimals=4))}', 'green'))

np.savetxt('Claculated test errors for correlation/Beijing/endpoints_errors.csv', endpoints_errors_100, 
           delimiter=',')
np.savetxt('Claculated test errors for correlation/Beijing/endpoints_stds.csv', endpoints_stds_100, 
           delimiter=',')


# # KNN for correlation matrix with 100 pairs

# In[68]:


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

from KNN_Class_Colab import KNN

metrics = ['discret_frechet', 'hausdorff', 'dtw', 'sspd', 'erp', 'edr', 'lcss', 
           fastdtw, dtw_tslearn, 'd_Q_pi']

path = 'Calculated Distance Matrices for KNN-Beijing-Correlation/'

train_test_mean_median_std_KNN_errors = []
test_KNN_errors = []
std_KNN_errors = []

for pair in pairs_100:
    print('pair =', pair)
    temp = []
    for i in range(len(metrics)):
        KNN_class = KNN(data[pair[0]], data[pair[1]], metric=metrics[i], gamma=1e-10, 
                        eps_edr=0.01, eps_lcss=0.01, Q_size=20, Q=None, p=2, 
                        path=path+str(metrics[i])+'-'+str(pair), 
                        test_size=0.3, n_neighbors=5, num_trials=100, pair=[0,1])

        KNN_class.write_matrix_to_csv()
        A = KNN_class.KNN_average_error()
        temp.append(A[1:])        
        print(i, A[0])
        print("=======================================================================")
    
    train_test_mean_median_std_KNN_errors.append(temp)
    test_KNN_errors.append(np.array(temp)[:,1])
    std_KNN_errors.append(np.array(temp)[:,3])
    print(colored("******************************************************************************", 'red'))
    print(colored("******************************************************************************", 'red'))
    print(colored("******************************************************************************", 'red'))
    
train_test_mean_median_std_KNN_errors = np.array(train_test_mean_median_std_KNN_errors)
test_KNN_errors = np.array(test_KNN_errors)
std_KNN_errors = np.array(std_KNN_errors)

list_test_error = list(np.round(np.mean(test_KNN_errors, 0), decimals=4))
list_std_error = list(np.round(np.mean(std_KNN_errors, 0), decimals=4))
print(colored(f'average test errors of 100 pairs: \n {list_test_error}', 'magenta'))
print(colored(f'average stds of 100 pairs: \n {list_std_error}', 'yellow'))


path = 'Claculated test errors for correlation/Beijing/KNN_test_errors_Beijing.csv'
np.savetxt(path, test_KNN_errors, delimiter=',')


# In[77]:


for i in range(100):
    path = 'Claculated test errors for correlation/Beijing/train_test_mean_median_std_KNN_errors/'+str(i)+'.csv'
    np.savetxt(path, train_test_mean_median_std_KNN_errors[i], delimiter=',')


# In[122]:


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
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean

from KNN_Class import KNN

path = 'Calculated Distance Matrices for KNN-Beijing-Correlation/'

train_test_mean_median_std_KNN_soft_dtw_errors = []
test_KNN_soft_dtw_errors = []
std_KNN_soft_dtw_errors = []
i = 0
for pair in pairs_100:
    st = time.time()
    print('pair =', pair)
    KNN_class = KNN(data[pair[0]], data[pair[1]], metric=SoftDTW, gamma=1e-15, 
                    eps_edr=0.01, eps_lcss=0.01, Q=None, p=2, 
                    path=path+'soft-dtw'+'-'+str(pair), 
                    test_size=0.3, n_neighbors=5, num_trials=100, pair=[0,1])

    KNN_class.write_matrix_to_csv()
    A = KNN_class.KNN_average_error()
    train_test_mean_median_std_KNN_soft_dtw_errors.append(A[1:])        
    print(i, A[0])
    i += 1
    print(colored(f"total time for pair {i}: {time.time()-st}", 'red'))
    print("=======================================================================")
    
train_test_mean_median_std_KNN_soft_dtw_errors = np.array(train_test_mean_median_std_KNN_soft_dtw_errors)
test_KNN_soft_dtw_errors = train_test_mean_median_std_KNN_soft_dtw_errors[:,1]
std_KNN_soft_dtw_errors = train_test_mean_median_std_KNN_soft_dtw_errors[:,3]

print(np.mean(test_KNN_soft_dtw_errors))

test_error = np.round(np.mean(test_KNN_soft_dtw_errors, 0), decimals=4)
std_error = np.round(np.mean(std_KNN_soft_dtw_errors, 0), decimals=4)
print(colored(f'average test errors of 100 pairs: {test_error}', 'magenta'))
print(colored(f'average stds of 100 pairs: {std_error}', 'yellow'))

path = 'Claculated test errors for correlation/Beijing/KNN_soft_dtw_test_errors_Beijing.csv'
np.savetxt(path, test_KNN_soft_dtw_errors, delimiter=',')


# In[124]:


path = 'Claculated test errors for correlation/Beijing/KNN_test_errors_Beijing.csv'
A = np.array(pd.read_csv(path, header=None))
path = 'Claculated test errors for correlation/Beijing/KNN_soft_dtw_test_errors_Beijing.csv'
B = np.array(pd.read_csv(path, header=None))
C = np.concatenate((A, B), 1)
print(C.shape)


# In[125]:


path = 'Claculated test errors for correlation/Beijing/train_test_mean_median_std_KNN_errors_11_distances.csv'
np.savetxt(path, C, delimiter=',')


# In[ ]:


path = 'Claculated test errors for correlation/Beijing/train_test_mean_median_std_KNN_errors_11_distances.csv'
KNN_test_errors = np.array(pd.read_csv(path, header=None)).T

path = 'Claculated test errors for correlation/Beijing/KNN_test_errors_Beijing.csv'
np.savetxt(path, KNN_test_errors, delimiter=',')


# ## KNN with LSH

# In[110]:


from KNN_with_LSH_class import KNN_with_LSH


# In[75]:


train_test_mean_median_std_KNN_LSH_errors = []
test_KNN_LSH_errors = []
std_KNN_LSH_errors = []
i = 0
for pair in pairs_100:
    st = time.time()
    print('pair =', pair)
    KNN_with_LSH_class = KNN_with_LSH(data[pair[0]], data[pair[1]], number_circles=20, 
                                      num_trials=100)
    A = KNN_with_LSH_class.KNN_LSH_average_error()
    train_test_mean_median_std_KNN_LSH_errors.append(A[1:])        
    print(i, A[0])
    i += 1
    print(colored(f"total time for pair {i}: {time.time()-st}", 'red'))
    print("=======================================================================")
    
train_test_mean_median_std_KNN_LSH_errors = np.array(train_test_mean_median_std_KNN_LSH_errors)
test_KNN_LSH_errors = train_test_mean_median_std_KNN_LSH_errors[:,1]
std_KNN_LSH_errors = train_test_mean_median_std_KNN_LSH_errors[:,3]

print(np.mean(test_KNN_LSH_errors))

test_error = np.round(np.mean(test_KNN_LSH_errors, 0), decimals=4)
std_error = np.round(np.mean(std_KNN_LSH_errors, 0), decimals=4)
print(colored(f'average test errors of 100 pairs: {test_error}', 'magenta'))
print(colored(f'average stds of 100 pairs: {std_error}', 'yellow'))

path = 'Claculated test errors for correlation/Beijing/KNN_LSH_test_errors_Beijing.csv'
np.savetxt(path, test_KNN_LSH_errors, delimiter=',')

path = 'Claculated test errors for correlation/Beijing/train_test_mean_median_std_KNN_LSH_errors.csv'
np.savetxt(path, train_test_mean_median_std_KNN_LSH_errors, delimiter=',')


# In[88]:


path = 'Claculated test errors for correlation/Beijing/KNN_test_errors_Beijing.csv'
A = np.array(pd.read_csv(path, header=None)).T

path = 'Claculated test errors for correlation/Beijing/KNN_LSH_test_errors_Beijing.csv'
B = np.array(pd.read_csv(path, header=None))

D = np.concatenate((A, B), 1)
print(D.shape)


# In[90]:


path = 'Claculated test errors for correlation/Beijing/train_test_mean_median_std_KNN_errors_12_distances.csv'
np.savetxt(path, D, delimiter=',')


# In[93]:


path = 'Claculated test errors for correlation/Beijing/train_test_mean_median_std_KNN_errors_12_distances.csv'
KNN_test_errors = np.array(pd.read_csv(path, header=None)).T

path = 'Claculated test errors for correlation/Beijing/KNN_test_errors_Beijing.csv'
np.savetxt(path, KNN_test_errors, delimiter=',')


# In[ ]:




