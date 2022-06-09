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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
import statsmodels.api as sm
from autograd import grad
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from termcolor import colored
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.svm import LinearSVC
from termcolor import colored
from datetime import datetime
import os
import similaritymeasures
import tslearn
from tslearn.metrics import dtw
from scipy.spatial.distance import directed_hausdorff
from frechetdist import frdist
import trjtrypy as tt
from trjtrypy.distances import d_Q
from trjtrypy.distances import d_Q_pi
import trjtrypy.visualizations as vs
from scipy.spatial import distance
from trjtrypy.featureMappings import curve2vec


# In[2]:


I = glob.glob('T-Drive/**/*.txt', recursive=True)


# In[3]:


# runtime about 35s
Start_time = time.time()
J = [0] * len(I)

for i in range(len(I)):
    for j in range(len(I)):
        if int(I[j][28:-4]) == i+1:
            J[i] = I[j]
print('total time =', time.time() - Start_time)


# In[4]:


taxi_id = np.array([J[i][28:-4] for i in range(len(J))])
I = np.sort(list(map(int, taxi_id)))
True in I == list(map(int, taxi_id)) # so the taxi ids are nn.arange(len(J))


# In[5]:


len(J)


# In[6]:


# create 'daysDate' function to convert start and end time to a float number of days

def days_date(time_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    current = datetime.strptime(time_str, date_format)
    date_format = "%Y-%m-%d"
    bench = datetime.strptime('2006-12-30', date_format)
    no_days = current - bench
    delta_time_days = no_days.days + current.hour / 24.0 + current.minute / (24. * 60.) + current.second / (24. * 3600.)
    return delta_time_days

days_date = np.vectorize(days_date)
float1 = np.vectorize(float)


# In[7]:


def read_file(file_name):
    data = []
    f = open(file_name, "r")
    for line in f:
        item = line.strip().split(",")
        if len(item) ==4:
            data.append(np.asarray(item[1:]))
    data = np.array(data)
    return data


# ## Remove stationary points

# In[8]:


def remove_segments(traj): # removes stationary points
    p2 = traj[:,1:][1:]
    p1 = traj[:,1:][:-1]
    L = ((p2-p1)*(p2-p1)).sum(axis =1)
    I = np.where(L>1e-16)[0]
    return traj[I]


# ### In data1 below:
#     (1) there is no stationary points
#     (2) there is no trajectories with less than 1 or more than 200 waypoints

# In[9]:


# runtime about 350s
Start_time = time.time()
T = len(J)
data1 = [] 
taxi_ids = []
for i in range(T):
    a = read_file(J[i])
    if len(a) > 1000:  
        a[:,0] = days_date(a[:,0])
        a = float1(a)
        data1.append(remove_segments(a)) 
        taxi_ids.append(i)

taxi_ids = np.array(taxi_ids)   
data1 = np.array(data1, dtype='object')
I = np.where(np.array([len(data1[i]) for i in range(len(data1))]) > 0)[0]
data1 = data1[I]
taxi_ids = np.array(taxi_ids[I], dtype='object')
print('total time =', time.time() - Start_time)
print("len(data1) =", len(data1))
print("selected taxi_ids: \n", taxi_ids)


# In[10]:


def partition(trajectory, threshold=20):
    trajectories = []
    a = 24 * 60 * (trajectory[-1][0] - trajectory[0][0])
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
                    val += 24 * 60 * (trajectory[:,0][1:][i] - trajectory[:,0][:-1][i])
                    i += 1
                else: 
                    break
            if len(trajectory[j:i-1]) > 0:
                trajectories.append(trajectory[j:i-1])
            a = a - val
        if len(trajectory[i:]) > 0:
            trajectories.append(trajectory[i:])
    trajectories = np.array(trajectories, dtype='object')
    return trajectories


# In[11]:


# Check to see if partitioning into less than 20 minutes worked correctly
for j in range(len(data1)):
    A = partition(data1[j], threshold=20)
    B = np.array([24 * 60 * (A[i][-1][0] - A[i][0][0]) for i in range(len(A))], dtype='object')
    I = np.where(B > 800)[0]
    if len(I) > 0: 
        print(j)


# ### In data3 below 
#     1) any taxi has between 100 and 200 trajectories
#     2) in each taxi dataset any trajectory has less than 20 minutes time

# In[12]:


Start_time = time.time()
data3 = []
idxs = []
for i in range(len(data1)):
    D = []
    A = partition(data1[i], threshold=20)
    for j in range(len(A)):
        if ((len(A[j]) >= 10) and (len(A[j]) <= 200)):
            D.append(A[j])
    if (len(D) >= 100 and len(D) <= 200):
        data3.append(np.array(D, dtype='object'))
        idxs.append(i)

taxi_idxs = taxi_ids[idxs]
data3 = np.array(data3, dtype='object')
taxi_idxs = np.array(taxi_idxs)

print('total time =', time.time() - Start_time)
print("len(data3) =", len(data3))
print("len(taxi_idxs) =", len(taxi_idxs))

A = np.array([len(data3[i]) for i in range(len(data3))])
print(colored(f"min and max length of selected users: {min(A), max(A)}", "green"))

print("number of selected users:", len(data3))
print("idxs: \n", idxs)
print(colored(f"selected taxi ids: \n {taxi_idxs}", "blue"))


# In[14]:


data = data3
Min = np.min([np.min([np.min(data[i][j][:,1:], 0) for j in range(len(data[i]))], 0) for i in range(len(data))], 0)
Max = np.max([np.max([np.max(data[i][j][:,1:], 0) for j in range(len(data[i]))], 0) for i in range(len(data))], 0)
Mean = np.mean([np.mean([np.mean(data[i][j][:,1:], 0) for j in range(len(data[i]))], 0) for i in range(len(data))], 0)
Std = np.std([np.std([np.std(data[i][j][:,1:], 0) for j in range(len(data[i]))], 0) for i in range(len(data))], 0)
Min, Mean, Max, Std


# In[15]:


pairs = []
for i in range(len(data)-1):
    for j in range(i+1, len(data)):
        pairs.append([i,j])
pairs = np.array(pairs)
print(len(pairs))


# In[16]:


#I = np.arange(len(pairs))
#np.random.shuffle(I)
#pairs_100 = pairs[I[:100]]
# we did above and chose the following pairs
pairs_100 = np.array([[ 87, 166], [161, 396], [370, 425], [209, 421], [ 84, 278], 
                      [ 24,  95], [ 72, 249], [358, 419], [ 86, 382], [241, 297],
                      [315, 363], [273, 335], [118, 270], [323, 430], [ 30, 431],
                      [111, 297], [ 17, 103], [191, 210], [179, 240], [ 53, 139],
                      [156, 240], [128, 412], [  8, 166], [ 96,  98], [270, 303],
                      [160, 319], [ 17,  41], [ 46, 381], [165, 216], [  8, 148],
                      [231, 287], [293, 426], [131, 264], [136, 271], [ 69, 137],
                      [120, 139], [111, 261], [ 13, 114], [138, 291], [ 31, 414], 
                      [252, 276], [ 38, 126], [ 31, 243], [382, 417], [240, 262],
                      [ 45,  49], [313, 359], [107, 206], [212, 243], [ 91, 383],
                      [118, 402], [ 31, 355], [ 74, 365], [110, 132], [136, 330],
                      [326, 376], [ 28, 128], [258, 302], [193, 256], [ 48, 151],
                      [255, 309], [340, 391], [ 90, 216], [341, 400], [ 45, 296], 
                      [206, 370], [184, 379], [248, 343], [ 16, 271], [245, 401],
                      [ 25, 130], [ 15, 430], [ 20, 285], [362, 385], [192, 343], 
                      [111, 410], [400, 414], [116, 185], [ 48, 202], [ 85, 234], 
                      [281, 392], [318, 320], [ 85, 417], [390, 427], [357, 428], 
                      [150, 189], [ 40, 343], [173, 249], [ 38, 399], [121, 393], 
                      [152, 316], [  0, 141], [ 24, 397], [255, 424], [ 92, 330], 
                      [121, 245], [ 78, 240], [ 68, 255], [ 21, 288], [278, 399]])


# In[18]:


taxi_ids_selected = list(set(pairs_100.reshape(200,)))
print("Chosen Users: \n", taxi_ids_selected)


# In[19]:


print("Chosen Users From Data:")
taxi_idxs[taxi_ids_selected] + 1


# In[20]:


data = data[taxi_ids_selected]
data.shape


# In[21]:


Dict = {taxi_ids_selected[i]:i for i in range(len(taxi_ids_selected))} 
pairs_new = []
for pair in pairs_100:
    pairs_new.append([Dict[pair[0]], Dict[pair[1]]])
pairs_new


# In[22]:


data_with_time = data + 0


# In[23]:


for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = data[i][j][:,1:]


# In[24]:


Min = np.min([np.min([np.min(data[i][j], 0) for j in range(len(data[i]))], 0) for i in range(len(data))], 0)
Max = np.max([np.max([np.max(data[i][j], 0) for j in range(len(data[i]))], 0) for i in range(len(data))], 0)
Mean = np.mean([np.mean([np.mean(data[i][j], 0) for j in range(len(data[i]))], 0) for i in range(len(data))], 0)
Std = np.std([np.std([np.std(data[i][j], 0) for j in range(len(data[i]))], 0) for i in range(len(data))], 0)
Min, Mean, Max, Std, LA.norm(Max - Mean)


# In[25]:


interesting_idx = []
for i in range(len(data)):
    I = np.where(np.array([np.min(data[i][j][:,0]) for j in range(len(data[i]))]) < 20)[0]
    if len(I) > 0:
        interesting_idx.append(i)
        
print(interesting_idx)


# In[26]:


print("Final Chosen Users From Data:")
taxi_idxs[taxi_ids_selected][interesting_idx] + 1


# In[27]:


for i in interesting_idx:
    print(i, set([len(data[i][j]) for j in range(len(data[i]))]))


# # Classifiers

# In[168]:


CC = [100, 100, 1000]
number_estimators = 50

clf1 = [make_pipeline(LinearSVC(dual=False, C=CC[0], tol=1e-5, 
                               class_weight ='balanced', max_iter=1000)), 
        "SVM, LinearSVC, C = "+str(CC[0])]
clf2 = [make_pipeline(StandardScaler(), svm.SVC(C=CC[1], kernel='rbf', gamma='auto', max_iter=200000)),
        "Gaussian SVM, C="+str(CC[1])+", gamma=auto"]
clf3 = [make_pipeline(StandardScaler(), svm.SVC(C=CC[2], kernel='poly', degree=3, max_iter=400000)),
        "Poly kernel SVM, C="+str(CC[2])+", deg=auto"]
clf4 = [DecisionTreeClassifier(), "Decision Tree"]
clf5 = [RandomForestClassifier(n_estimators=number_estimators), 
         "RandomForestClassifier, n="+str(number_estimators)]
clf6 = [KNeighborsClassifier(n_neighbors=5), "KNN"]
clf7 = [LogisticRegression(solver='lbfgs'), "Logistic Regression"]

clf = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]

classifs = [item[0] for item in clf]
keys = [item[1] for item in clf]


# In[27]:


pairs_errors = []
i = 0
for pair in pairs_new:
    A = classification_without_exp_random_Q(data[pair[0]], data[pair[1]], Q_size=20, 
                                            epoch=100, classifiers=[clf7])
    pairs_errors.append([pair, np.round(A[-1][0], decimals=2)])
    print('i =', i)
    print(colored(f"pair={pair}", 'green'))
    print(A[0])
    print("===================================================================")
    i += 1


# In[32]:


print(f"selected 1oo random pairs of taxies in T-Drive dataset: \n {pairs_100.tolist()}")
print(f"selected random 100 pairs of taxies in data dataset: \n {pairs_new}")
#selected_pairs_of_taxies = 


# ### The following two boxes are for above 0.20 test error that newly are done

# In[38]:


pairs_100 = np.array(pairs_new, dtype='object') + 0


# # Generating distance matrices

# ## $d_Q^{\pi}$

# In[85]:


def calculate_dists_d_Q_pi(data1, data2, p, path): 
    start_time = time.time() 
    data = np.concatenate((data1, data2), 0)
    n = len(data)
    A = []
    for i in range(n-1):
        for j in range(i+1, n):
            A.append(d_Q_pi(Q, data[i], data[j], p=p))
    A = np.array(A)
    tri = np.zeros((n, n))
    tri[np.triu_indices(n, 1)] = A
    for i in range(1, n):
        for j in range(i):
            tri[i][j] = tri[j][i]
    np.savetxt(path, tri, delimiter=',')

    total_time = time.time() - start_time
    return total_time


# In[ ]:


# the following path is for 100 chosen pairs for correlation and R^2 test
paths = []
for i in range(len(pairs_100)): 
    paths.append('Calculated Distance Matrices for KNN for comparison with RF/Taxi/d_Q_pi/Taxi-Pairs['+str(pairs_100[i])+']-d_Q_pi.csv')

for i in range(len(pairs_100)):
    calculate_dists_d_Q_pi(data[pairs_100[i][0]], data[pairs_100[i][1]], p=1, path=paths[i])


# In[40]:


def KNN_with_d_Q_pi(n_1, n_2, dist_matrix):
    
    '''path example: '/Users/hasan/Desktop/Anaconda/Research/Calculated Distance Matrices for KNN/Taxi-Pairs['+str(pairs_final[i])+']-d_Q_pi.csv'
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


# In[41]:


def KNN_average_error_d_Q_pi(data1, data2, num_trials, path_to_dists, pair):

    '''path_to_dists: the path to the corresponding distance matrix'''

    Start_time = time.time()

    train_errors = np.zeros(num_trials)
    test_errors = np.zeros(num_trials)
    
    dist_matrix = np.array(pd.read_csv(path_to_dists, header=None))

    for i in range(num_trials):
        train_errors[i], test_errors[i] = KNN_with_d_Q_pi(len(data1), len(data2), dist_matrix)

    Dict = {}
    Dict[1] = [f"KNN with d_Q_pi for pairs {pair}", 
                    np.round(np.mean(train_errors), decimals = 4), 
                    np.round(np.mean(test_errors), decimals = 4), 
                    np.round(np.std(test_errors), decimals = 4)]

    df = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier',
                                'Train Error', 'Test Error', 'std'])
    print(colored(f"num_trials = {num_trials}", "blue"))
    print(colored(f'total time = {time.time() - Start_time}', 'green'))

    return (df, np.mean(train_errors), np.mean(test_errors), np.median(test_errors), 
            np.std(test_errors))


# In[ ]:


## the following path is for 100 chosen pairs for correlation and R^2 test
paths = []
for i in range(len(pairs_100)):
    paths.append('/Users/hasan/Desktop/Anaconda/Research/Calculated Distance Matrices for KNN for comparison with RF/Taxi/d_Q_pi/Taxi-Pairs['+str(pairs_100[i])+']-d_Q_pi.csv')

train_test_median_std_errors_d_Q_pi = []
for i in range(len(pairs_100)):
    print("i=", i)
    print(colored(f"pair: {pairs_100[i]}", 'magenta'))
    A = KNN_average_error_d_Q_pi(data[pairs_100[i][0]], data[pairs_100[i][1]], 
                                 num_trials=100, path_to_dists=paths[i], pair=pairs_100[i])
    train_test_median_std_errors_d_Q_pi.append(A[1:])
    print(A[0])
    print("=======================================================================")


# In[679]:


path = '/Users/hasan/Desktop/Anaconda/Research/Calculated KNN errors for comparison with RF/Taxi-Pairs-d_Q_pi.csv'
np.savetxt(path, np.array(train_test_median_std_errors_d_Q_pi), delimiter=',')


# ## dtw from tslearn

# In[42]:


def calculate_dists_dtw_tslearn(data1, data2, path): 
    start_time = time.time() 
    data = np.concatenate((data1, data2), 0)
    n = len(data)
    A = []
    for i in range(n-1):
        for j in range(i+1, n):
            A.append(tslearn.metrics.dtw(data[i], data[j]))
    A = np.array(A)
    tri = np.zeros((n, n))
    tri[np.triu_indices(n, 1)] = A
    for i in range(1, n):
        for j in range(i):
            tri[i][j] = tri[j][i]
    np.savetxt(path, tri, delimiter=',')

    total_time = time.time() - start_time
    return total_time


# In[180]:


# the following path is for 100 chosen pairs for correlation and R^2 test
paths = []
for i in range(len(pairs_100)): 
    paths.append('/Users/hasan/Desktop/Anaconda/Research/Calculated Distance Matrices for KNN for comparison with RF/dtw_tslearn/Taxi/Taxi-Pairs['+str(pairs_100[i])+']-dtw-tslearn.csv')

for i in range(len(pairs_100)):
    calculate_dists_dtw_tslearn(data[pairs_100[i][0]], data[pairs_100[i][1]], path=paths[i])
    


# In[43]:


def KNN_with_dists_dtw_tslearn(n_1, n_2, path_to_dists):
    '''path example: '/Users/hasan/Desktop/Anaconda/Research/Calculated Distance Matrices for KNN/Taxi-Pairs['+str(pairs_final[i])+']-dtw-tslearn.csv'
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


# In[44]:


def KNN_average_error_dtw_tslearn(data1, data2, num_trials, path_to_dists, pair):

    '''path_to_dists: the path to the corresponding distance matrix'''

    Start_time = time.time()

    train_errors = np.zeros(num_trials)
    test_errors = np.zeros(num_trials)

    for i in range(num_trials):
        train_errors[i], test_errors[i] = KNN_with_dists_dtw_tslearn(len(data1), len(data2), path_to_dists)

    Dict = {}
    Dict[1] = [f"KNN with dtw from tslearn for pairs {pair}", 
                    np.round(np.mean(train_errors), decimals = 4), 
                    np.round(np.mean(test_errors), decimals = 4), 
                    np.round(np.std(test_errors), decimals = 4)]

    df = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier',
                                'Train Error', 'Test Error', 'std'])
    print(colored(f"num_trials = {num_trials}", "blue"))
    print(colored(f'total time = {time.time() - Start_time}', 'green'))

    return (df, np.mean(train_errors), np.mean(test_errors), np.median(test_errors), 
            np.std(test_errors))


# In[ ]:


# # the following path is for 100 chosen pairs for correlation and R^2 test
paths = []
for i in range(len(pairs_100)):
    paths.append('/Users/hasan/Desktop/Anaconda/Research/Calculated Distance Matrices for KNN for comparison with RF/Taxi/dtw_tslearn/Taxi-Pairs['+str(pairs_100[i])+']-dtw-tslearn.csv')

train_test_median_std_errors_dtw_tslearn = []
for i in range(len(pairs_100)):
    print("i=", i)
    print(colored(f"pair: {pairs_100[i]}", 'magenta'))
    A = KNN_average_error_dtw_tslearn(data[pairs_100[i][0]], data[pairs_100[i][1]], num_trials=100, 
                            path_to_dists=paths[i], pair=pairs_100[i])
    train_test_median_std_errors_dtw_tslearn.append(A[1:])
    print(A[0])
    print("=====================================================================")


# In[364]:


path = '/Users/hasan/Desktop/Anaconda/Research/Calculated KNN errors for comparison with RF/Taxi-Pairs-dtw_tslearn.csv'
np.savetxt(path, np.array(train_test_median_std_errors_dtw_tslearn), delimiter=',')


# ## From github page: https://github.com/bguillouet/traj-dist
# 
# It includes 9 distances for trajectories including: Continuous Frechet, Discrete Frechet, Hausdorff, DTW, SSPD, LCSS, EDR, ERP.
# 
# ## All but the continuous Frechet distance are really fast.

# In[301]:


import traj_dist.distance as tdist
import pickle


# In[45]:


def KNN_with_dists(n_1, n_2, dists_names, paths_to_dists):
    '''path example: '/content/gdrive/My Drive/traj-dist/Calculated Distance Matrices (car-bus)/sspd.csv'
       dists_names: a list of distance names
       paths_to_dists: the paths list to the corresponding distancas (each path 
                       points out to the corresponding distance matrix)
       n_1: len(data_1)
       n_2: len(data_2)
       dist_name: the name of distance used to calculate sitance matrix 
       (the name is taken from a list above called metrics)'''

    train_errors = np.zeros(len(dists_names))
    test_errors = np.zeros(len(dists_names))

    I_1, J_1, y_train_1, y_test_1 = train_test_split(np.arange(n_1), 
                                                np.ones(n_1), test_size=0.3)
    I_2, J_2, y_train_2, y_test_2 = train_test_split(np.arange(n_1, n_1+n_2), 
                                                np.ones(n_2), test_size=0.3)
    labels = np.array([1]*n_1 + [0] * n_2)
    I = np.concatenate((I_1, I_2), 0)
    np.random.shuffle(I)
    J = np.concatenate((J_1, J_2), 0)
    np.random.shuffle(J)

    for i in range(len(dists_names)):

        dist_matrix = np.array(pd.read_csv(paths_to_dists[i], header=None))

        D_train = dist_matrix[I][:, I]
        D_test = dist_matrix[J][:,I]
        train_labels = labels[I]
        test_labels = labels[J]

        clf = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
        
        #Train the model using the training sets
        clf.fit(D_train, list(train_labels))

        #Predict labels for train dataset
        train_pred = clf.predict(D_train)
        train_errors[i] = sum(train_labels != train_pred)/len(I)
        
        #Predict labels for test dataset
        test_pred = clf.predict(D_test)
        test_errors[i] = sum((test_labels != test_pred))/len(J)
        
    return train_errors, test_errors


# In[46]:


def KNN_average_error_7_dists(data1, data2, num_trials, dists_names, paths_to_dists):

    '''dists_names: a list of distance names
       paths_to_dists: the paths list to the corresponding distancas (each path 
                       points out to the corresponding distance matrix)'''

    Start_time = time.time()

    train_errors = np.zeros((num_trials, len(dists_names)))
    test_errors = np.zeros((num_trials, len(dists_names)))

    for i in range(num_trials):
        tr_errors, ts_errors = KNN_with_dists(len(data1), len(data2), dists_names, paths_to_dists)
        train_errors[i] = tr_errors
        test_errors[i] = ts_errors

    train_error = np.mean(train_errors, axis=0)
    test_error = np.mean(test_errors, axis=0)
    std_test_error = np.std(test_errors, axis=0)

    Dict = {}
    for i in range(len(dists_names)):
        Dict[i+1] = [f"KNN with {dists_names[i]}", 
                     np.round(train_error[i], decimals = 4), 
                     np.round(test_error[i], decimals = 4), 
                     np.round(std_test_error[i], decimals = 4)]

    df = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier',
                                'Train Error', 'Test Error', 'std'])
    print(colored(f"num_trials = {num_trials}", "blue"))
    print(colored(f'total time = {time.time() - Start_time}', 'green'))

    return [df, train_errors, test_errors, train_error, test_error, 
            np.median(test_errors, axis=0), np.std(test_errors, axis=0)]


# In[302]:


# For 100 pairs
Metrics = ['discret_frechet', 'hausdorff', 'dtw', 'sspd', 'lcss', 'edr', 'erp']

#tdist.pdist(data[pairs_100[5][1]][:10], 'edr', eps=1e-10) 
#tdist.pdist(data[pairs_100[5][1]][:10], 'lcss', eps=0.05) 

for i in range(len(pairs_100)):
    st_time = time.time()
    calculate_dists(data, metrics=Metrics, pair=pairs_100[i], eps_edr=1e-10, eps_lcss=0.05)
    print(f"time for {i}-th pair: {time.time() - st_time}")


# In[304]:


# the following path is for 100 chosen pairs for correlation and R^2 test
paths = []
for i in range(len(pairs_100)):
    paths_temp = []
    for j in range(len(Metrics)):
        paths_temp.append('Calculated Distance Matrices for KNN for comparison with RF/Taxi/7-dists/Taxi-Pairs'+str(pairs_100[i])+'-'+Metrics[j]+'.csv')
    paths.append(paths_temp)
    
train_test_median_std_errors_7_dists = []
for i in range(len(pairs_100)):
    print("i=", i)
    A = KNN_average_error_7_dists(data[pairs_100[i][0]], data[pairs_100[i][1]], 
                          num_trials=100, dists_names=Metrics, paths_to_dists=paths[i])
    train_test_median_std_errors_7_dists.append(A[3:])
    print(colored(f"pair: {pairs_100[i]}", 'magenta'))
    print(A[0])
    print("============================================================")


# In[305]:


train_test_median_std_errors_discret_frechet = np.array(train_test_median_std_errors_7_dists)[:,:,0]
train_test_median_std_errors_hausdorff = np.array(train_test_median_std_errors_7_dists)[:,:,1]
train_test_median_std_errors_dtw = np.array(train_test_median_std_errors_7_dists)[:,:,2]
train_test_median_std_errors_sspd = np.array(train_test_median_std_errors_7_dists)[:,:,3]
train_test_median_std_errors_lcss = np.array(train_test_median_std_errors_7_dists)[:,:,4]
train_test_median_std_errors_edr = np.array(train_test_median_std_errors_7_dists)[:,:,5]
train_test_median_std_errors_erp = np.array(train_test_median_std_errors_7_dists)[:,:,6]


# In[306]:


for i in range(len(Metrics)):
    path = '/Users/hasan/Desktop/Anaconda/Research/Calculated KNN errors for comparison with RF/Taxi-Pairs'+'-'+Metrics[i]+'.csv'
    np.savetxt(path, np.array(train_test_median_std_errors_7_dists)[:,:,i], delimiter=',')


# ## fastdtw which is an approximate dtw

# In[77]:


from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def calculate_fastdtw_dists(data1, data2, path): 
    start_time = time.time() 
    X = np.concatenate((data1, data2), 0)
    n = len(X)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i][i:] = [fastdtw(X[i], X[j])[0] for j in range(i, n)]
    for i in range(1, n):
        for j in range(i):
            matrix[i][j] = matrix[j][i]
    np.savetxt(path, matrix, delimiter=',')
    total_time = time.time() - start_time
    return total_time


# In[273]:


# the following path is for 100 chosen pairs for correlation and R^2 test
paths = []
for i in range(len(pairs_100)): 
    paths.append('/Users/hasan/Desktop/Anaconda/Research/Calculated Distance Matrices for KNN for comparison with RF/Taxi/fastdtw/Taxi/Taxi-Pairs['+str(pairs_100[i])+']-fastdtw.csv')

for i in range(len(pairs_100)):
    calculate_fastdtw_dists(data[pairs_100[i][0]], data[pairs_100[i][1]], path=paths[i])
    


# In[66]:


def KNN_with_fastdtw(n_1, n_2, path_to_dists):
    '''n_1: len(data_1)
       n_2: len(data_2)
       dist_name: the name of distance used to calculate sitance matrix 
       (the name is taken from a list above called metrics)'''

    I_1, J_1, y_train_1, y_test_1 = train_test_split(np.arange(n_1), 
                                                np.ones(n_1), test_size=0.3)
    I_2, J_2, y_train_2, y_test_2 = train_test_split(np.arange(n_1, n_1+n_2), 
                                                np.ones(n_2), test_size=0.3)
    labels = np.array([1]*n_1 + [0] * n_2)
    I = np.concatenate((I_1, I_2), 0)
    np.random.shuffle(I)
    J = np.concatenate((J_1, J_2), 0)
    np.random.shuffle(J)
    dist_matrix = np.array(pd.read_csv(path_to_dists, header=None))

    D_train = dist_matrix[I][:, I]
    D_test = dist_matrix[J][:,I]
    train_labels = labels[I]
    test_labels = labels[J]

    clf = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
    
    #Train the model using the training sets
    clf.fit(D_train, list(train_labels))

    #Predict labels for train dataset
    train_pred = clf.predict(D_train)
    train_errors = sum(train_labels != train_pred)/len(I)
    
    #Predict labels for test dataset
    test_pred = clf.predict(D_test)
    test_errors = sum((test_labels != test_pred))/len(J)
        
    return train_errors, test_errors


# In[67]:


def KNN_fastdtw_average_error(data1, data2, num_trials, path_to_dists, pair):

    '''dists_names: a list of distance names
       paths_to_dists: the paths list to the corresponding distancas (each path 
                       points out to the corresponding distance matrix)'''

    Start_time = time.time()

    train_errors = np.zeros(num_trials)
    test_errors = np.zeros(num_trials)

    for i in range(num_trials):
        tr_errors, ts_errors = KNN_with_fastdtw(len(data1), len(data2), path_to_dists)
        train_errors[i] = tr_errors
        test_errors[i] = ts_errors

    train_error = np.mean(train_errors)
    test_error = np.mean(test_errors)
    std_test_error = np.std(test_errors)

    Dict = {}
    Dict[1] = [f"KNN with fastdtw pair {pair}", 
               np.round(train_error, decimals = 4), 
               np.round(test_error, decimals = 4), 
               np.round(std_test_error, decimals = 4)]

    df = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier',
                                'Train Error', 'Test Error', 'std'])
    print(colored(f"num_trials = {num_trials}", "blue"))
    print(colored(f'total time = {time.time() - Start_time}', 'green'))

    return (df, train_errors, test_errors, train_error, test_error, 
            np.median(test_errors), np.std(test_errors))


# In[ ]:


# the following path is for 100 chosen pairs for correlation and R^2 test
paths = []
for i in range(len(pairs_100)):
    paths.append('Calculated Distance Matrices for KNN for comparison with RF/Taxi/fastdtw/Taxi-Pairs['+str(pairs_100[i])+']-fastdtw.csv')

train_test_median_std_errors_fastdtw = []
for i in range(len(pairs_100)):
    print(colored(f"i = {i}, pair: {pairs_100[i]}", 'magenta'))
    A = KNN_fastdtw_average_error(data[pairs_100[i][0]], data[pairs_100[i][1]], 
                                  num_trials=100, path_to_dists=paths[i], pair=pairs_100[i])
    train_test_median_std_errors_fastdtw.append(A[3:])
    print(A[0])
    print("=====================================================================")


# In[284]:


path = '/Users/hasan/Desktop/Anaconda/Research/Calculated KNN errors for comparison with RF/Taxi-Pairs-fastdtw.csv'
np.savetxt(path, np.array(train_test_median_std_errors_fastdtw), delimiter=',')


# ## Soft-dtw

# In[43]:


from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean

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


# In[ ]:


# the following path is for 100 chosen pairs for correlation and R^2 test
paths = []
for i in range(len(pairs_100)): 
    paths.append('Calculated Distance Matrices for KNN for comparison with RF/Taxi/soft-dtw/Taxi/Taxi-Pairs['+str(pairs_100[i])+']-soft-dtw.csv')

for i in range(len(pairs_100)):
    calculate_dists_soft_dtw(data[pairs_100[i][0]], data[pairs_100[i][1]], 
                             gamma=1e-14, path=paths[i])
    


# In[45]:


def KNN_with_dists_soft_dtw(n_1, n_2, path_to_dists):
    '''path example: path = '/Users/hasan/Desktop/Anaconda/Research/Calculated Distance Matrices for KNN/Characters-Pairs['+str(i)+']-soft-dtw.csv'
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


# In[46]:


def KNN_average_error_soft_dtw(data1, data2, num_trials, path_to_dists, pair):

    '''path_to_dists: the path to the corresponding distance matrix'''

    Start_time = time.time()

    train_errors = np.zeros(num_trials)
    test_errors = np.zeros(num_trials)

    for i in range(num_trials):
        train_errors[i], test_errors[i] = KNN_with_dists_soft_dtw(len(data1), len(data2), path_to_dists)

    Dict = {}
    Dict[1] = [f"KNN with soft-dtw for pairs {pair}", 
                    np.round(np.mean(train_errors), decimals = 4), 
                    np.round(np.mean(test_errors), decimals = 4), 
                    np.round(np.std(test_errors), decimals = 4)]

    df = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier',
                                'Train Error', 'Test Error', 'std'])
    print(colored(f"num_trials = {num_trials}", "blue"))
    print(colored(f'total time = {time.time() - Start_time}', 'green'))

    return (df, np.mean(train_errors), np.mean(test_errors), np.median(test_errors), 
            np.std(test_errors))


# In[ ]:


# the following path is for 100 chosen pairs for correlation and R^2 test
paths = []
for i in range(len(pairs_100)):
    paths.append('Calculated Distance Matrices for KNN for comparison with RF/Taxi/soft-dtw/Taxi-Pairs['+str(pairs_100[i])+']-soft-dtw.csv')

train_test_median_std_errors_soft_dtw = []
for i in range(len(pairs_100)):
    print("i=", i)
    print(colored(f"pair: {pairs_100[i]}", 'magenta'))
    A = KNN_average_error_soft_dtw(data[pairs_100[i][0]], data[pairs_100[i][1]], 
                                              num_trials=100, path_to_dists=paths[i], 
                                              pair=pairs_100[i])
    train_test_median_std_errors_soft_dtw.append(A[1:])
    print(A[0])
    print("=====================================================================")


# In[300]:


path = '/Users/hasan/Desktop/Anaconda/Research/Calculated KNN errors for comparison with RF/Taxi-Pairs-soft_dtw.csv'
np.savetxt(path, np.array(train_test_median_std_errors_soft_dtw), delimiter=',')


# # Correlation

# In[31]:


pairs_100 = pairs_new


# ## Classification of pairs_100 with $v_Q$ with random $Q$ in each iteration

# In[327]:


clfs = clff[1:2] + clff[6:7] + clff[9:12] + clff[:1] + clff[13:]
v_Q_errors_100 = []
v_Q_stds_100 = []

for pair in pairs_100:
    print(colored(f"pair={pair}", 'green'))
    classif = binaryClassificationAverageMajority(data[pair[0]], data[pair[1]], Q_size=20,
                                            epoch=100, num_trials_maj=1, classifiers=clfs, 
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

np.savetxt('Claculated test errors for correlation/Taxi/random_V_Q_errors.csv', v_Q_errors_100, 
           delimiter=',')
np.savetxt('Claculated test errors for correlation/Taxi/random_V_Q_stds.csv', v_Q_stds_100, 
           delimiter=',')


# In[57]:


pd.read_csv('Claculated test errors for correlation/Taxi/random_V_Q_errors.csv', 
            header=None).mean(0)


# ## Classification for pairs_100 with $v_Q^{\sigma}$ with random $Q$ in each iteration

# In[52]:


clfs = clff[1:2] + clff[6:7] + clff[9:12] + clff[:1] + clff[13:]
v_Q_sigma_errors_100 = []
v_Q_sigma_stds_100 = []

for pair in pairs_100:
    print(colored(f"pair={pair}", 'green'))
    classif = binaryClassificationAverageMajority(data[pair[0]], data[pair[1]], Q_size=20,
                                            epoch=100, num_trials_maj=1, classifiers=clfs, 
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

np.savetxt('Claculated test errors for correlation/Taxi/random_V_Q_sigma_errors.csv', 
           v_Q_sigma_errors_100, delimiter=',')
np.savetxt('Claculated test errors for correlation/Taxi/random_V_Q_sigma_stds.csv', 
           v_Q_sigma_stds_100, delimiter=',')


# ## Classification for pairs_100 with endpoints

# In[58]:


clfs = clff[1:2] + clff[6:7] + clff[9:12] + clff[:1] + clff[13:]
endpoints_errors_100 = []
endpoints_stds_100 = []

for pair in pairs_100:
    print(colored(f"pair={pair}", 'green'))
    classif = binaryClassificationAverageMajority(data[pair[0]], data[pair[1]], Q_size=20,
                                            epoch=100, num_trials_maj=1, classifiers=clfs, 
                                            version='unsigned', test_size=0.3)
    A = classif.endpoint_classification()
    print(A[0])
    endpoints_errors_100.append(np.round(A[2], decimals=4).tolist())
    endpoints_stds_100.append(np.round(A[3], decimals=4).tolist())
    print("===========================================================================")

print('endpoints_errors_100 = \n', endpoints_errors_100)
print('endpoints_stds_100 = \n', endpoints_stds_100)
print("===========================================================================")
print(colored(f'average errors: {list(np.round(np.mean(endpoints_errors_100, 0), decimals=4))}', 'blue'))
print(colored(f'average stds: {list(np.round(np.mean(endpoints_stds_100, 0), decimals=4))}', 'green'))

np.savetxt('Claculated test errors for correlation/Taxi/endpoints_errors.csv', endpoints_errors_100, 
           delimiter=',')
np.savetxt('Claculated test errors for correlation/Taxi/endpoints_stds.csv', endpoints_stds_100, 
           delimiter=',')


# ## KNN with LSH for pairs_100

# In[32]:


from KNN_with_LSH_class import KNN_with_LSH


# In[36]:


train_test_median_std_errors_LSH = []

for i in range(len(pairs_100)):
    st = time.time()
    print(colored(f"i = {i}, pair: {pairs_100[i]}", 'magenta'))
    KNN_with_LSH_class = KNN_with_LSH(data[pairs_100[i][0]], data[pairs_100[i][1]], 
                                      number_circles=20, radius=0.1, num_trials=100)
    A = KNN_with_LSH_class.KNN_LSH_average_error()
    train_test_median_std_errors_LSH.append(A[1:]) 
    print(A[0])
    print(colored(f"total time for pair {i}: {time.time()-st}", 'red'))
    print("====================================================")

train_test_median_std_errors_LSH = np.array(train_test_median_std_errors_LSH)
test_KNN_LSH_errors = train_test_median_std_errors_LSH[:,1]
std_KNN_LSH_errors = train_test_median_std_errors_LSH[:,3]

print(np.mean(test_KNN_LSH_errors))

test_error = np.round(np.mean(test_KNN_LSH_errors, 0), decimals=4)
std_error = np.round(np.mean(std_KNN_LSH_errors, 0), decimals=4)
print(colored(f'average test errors of 100 pairs: {test_error}', 'magenta'))
print(colored(f'average stds of 100 pairs: {std_error}', 'yellow'))

path = 'Claculated test errors for correlation/Taxi/KNN_LSH_test_errors_Taxi.csv'
np.savetxt(path, test_KNN_LSH_errors, delimiter=',')
    
path = 'Calculated KNN errors for comparison with RF/Taxi-Pairs-LSH.csv'
np.savetxt(path, train_test_median_std_errors_LSH, delimiter=',')


# ## Correlation Matrix Plot

# In[718]:


# Extra
distances = ['discret_frechet', 'hausdorff', 'dtw', 'soft_dtw', 'fastdtw', 'dtw_tslearn',
             'd_Q_pi', 'sspd', 'erp', 'lcss', 'edr']

corr_values = []
R_squared_values = []
KNN_test_errors = []

for i in range(len(distances)):
    path = 'Calculated KNN errors for comparison with RF/Taxi-Pairs-'+distances[i]+'.csv'
    A = np.array(pd.read_csv(path, header=None))
    corr_matrix = np.corrcoef(test_errors_RF, A[:,1])
    corr = corr_matrix[0,1]
    R_sq = corr**2
    
    corr_values.append(np.cov([np.array(test_errors_RF), A[:,1]])[0,1])
    R_squared_values.append(R_sq)
    KNN_test_errors.append(A[:,1])
    
path = 'Claculated test errors for correlation/Taxi/KNN_test_errors.csv'
#np.savetxt(path, KNN_test_errors, delimiter=',')


# In[60]:


path2 = 'Claculated test errors for correlation/Taxi/KNN_test_errors.csv'
KNN_test_errors_11 = np.array(pd.read_csv(path2, header=None))

path3 = 'Claculated test errors for correlation/Taxi/KNN_LSH_test_errors_Taxi.csv'
KNN_LSH_test_errors = np.array(pd.read_csv(path3, header=None)).T

KNN_test_errors = np.concatenate((KNN_test_errors_11, KNN_LSH_test_errors), 0)

path = 'Claculated test errors for correlation/Taxi/KNN_test_errors_12.csv'
np.savetxt(path, KNN_test_errors, delimiter=',')


# In[58]:


# This one should be used
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.cluster import SpectralCoclustering

distances = ['discret_frechet', 'hausdorff', 'dtw', 'soft_dtw', 'fastdtw', 'dtw_tslearn',
             'd_Q_pi', 'sspd', 'erp', 'lcss', 'edr', 'LSH']

classifiers = ['LSVM', 'GSVM', 'PSVM', 'DT', 'RF', 'KNN', 'LR', 'Prn']

path1 = 'Claculated test errors for correlation/Taxi/random_V_Q_errors.csv'
classifiers_errors = np.array(pd.read_csv(path1, header=None))

path2 = 'Claculated test errors for correlation/Taxi/KNN_test_errors.csv'
KNN_test_errors_11 = np.array(pd.read_csv(path2, header=None))

path3 = 'Claculated test errors for correlation/Taxi/KNN_LSH_test_errors_Taxi.csv'
KNN_LSH_test_errors = np.array(pd.read_csv(path3, header=None)).T

KNN_test_errors = np.concatenate((KNN_test_errors_11, KNN_LSH_test_errors), 0)

dataa = {distances[i]: KNN_test_errors[i] for i in range(len(distances))}
dataa1 = {classifiers[i]: classifiers_errors[:, i] for i in range(len(classifiers))}
dataa.update(dataa1)

columns = [distances[i] for i in range(len(distances))] + [classifiers[i] for i in range(len(classifiers))]
df = pd.DataFrame(dataa, columns=columns)
corrMatrix = df.corr()

plt.figure(figsize = (8, 6))
sn.heatmap(corrMatrix, annot=False, cmap='RdBu', center=0)
plt.title(f'Original Correlation Matrix')
#plt.savefig(f'Original-Correlation-Matrix.png', bbox_inches='tight', dpi=1200)
plt.show()


model = SpectralCoclustering(n_clusters=5, random_state=None)
model.fit(corrMatrix)

columns = np.array(distances + classifiers)[np.argsort(model.column_labels_)]

keys = []
values = []
for i in np.argsort(model.column_labels_):
    if i < len(distances):
        keys.append(distances[i])
        values.append(KNN_test_errors[i])
    else:
        keys.append(classifiers[i-len(distances)])
        values.append(classifiers_errors[:, i-len(distances)])
        
Dataa = {keys[i]: values[i] for i in range(len(keys))}
ddf = pd.DataFrame(Dataa, columns=columns)

plt.figure(figsize = (8, 6))
sn.heatmap(ddf.corr(), annot=False, cmap='RdBu', center=0)
plt.title(f'Clustered Correlation Matrix (4 Clusters)')
#plt.savefig(f'Clustered-Correlation-Matrix.png', bbox_inches='tight', dpi=1200)
plt.show()


# In[59]:


corrMatrix


# In[ ]:




