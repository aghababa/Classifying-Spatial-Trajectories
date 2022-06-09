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


# In[13]:


taxi_idxs + 1


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


# In[17]:


print(pairs_100.tolist())


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


# In[33]:


def classification_without_exp_random_Q(data_1, data_2, Q_size, epoch, classifiers):
    start_time = time.time()
    models = [item[0] for item in classifiers]
    keys = [item[1] for item in classifiers]

    r = len(classifiers)

    train_error_mean = np.zeros(r)
    test_error_mean = np.zeros(r)
    test_error_std = np.zeros(r)
    
    train_errors = np.zeros((r, epoch)) 
    test_errors = np.zeros((r, epoch))

    n_1 = len(data_1)
    n_2 = len(data_2) 

    for s in range(epoch):
        I1, I2, I3, I4 = train_test_split(np.arange(n_1), [1] * n_1, test_size=0.3) 
        J1, J2, J3, J4 = train_test_split(np.arange(n_2), [-1] * n_2, test_size=0.3)

        x_preds = np.zeros((r, len(I1) + len(J1)))
        y_preds = np.zeros((r, len(I2) + len(J2)))

        train = np.concatenate((data_1[I1], data_2[J1]), 0)
        test = np.concatenate((data_1[I2], data_2[J2]), 0)

        Min = np.min([np.min(train[i], 0) for i in range(len(train))], 0)
        Max = np.max([np.max(train[i], 0) for i in range(len(train))], 0)
        Mean = np.mean([np.mean(train[i], 0) for i in range(len(train))], 0)
        Std = np.std([np.std(train[i], 0) for i in range(len(train))], 0)
       
        Q = np.ones((Q_size, 2))
        Q[:,0] = np.random.normal(Mean[0], 4 * Std[0], Q_size)
        Q[:,1] = np.random.normal(Mean[1], 4 * Std[1], Q_size)

        X = curve2vec(Q, train)
        y = np.concatenate((I3, J3), axis = 0)

        test_data = curve2vec(Q, test)
        test_labels = np.concatenate((I4, J4), axis = 0)

        for k in range(r): 
            model = models[k]
            model.fit(X, y)

            x_preds[k] = model.predict(X)                
            y_preds[k] = model.predict(test_data)
        
        for k in range(r):
            train_errors[k][s] = 1 - metrics.accuracy_score(y, x_preds[k])
            test_errors[k][s] = 1 - metrics.accuracy_score(test_labels, y_preds[k])
            
    for k in range(r):
        train_error_mean[k] = np.mean(train_errors[k])
        test_error_mean[k] = np.mean(test_errors[k])
        test_error_std[k] = np.std(test_errors[k])
    
    Dict = {}

    for k in range(len(keys)): 
        Dict[k+1] = [keys[k], np.round(train_error_mean[k], decimals = 4), 
                     np.round(test_error_mean[k], decimals = 4),
                     np.round(test_error_std[k], decimals = 4)]

    pdf = pd.DataFrame.from_dict(Dict, orient='index', 
                columns=['Classifier','Train Error', 'Test Error', 'Std Error'])
    
    print(colored(f"total time = {time.time() - start_time}", "red"))

    return pdf, train_error_mean, test_error_mean


# In[29]:


pairs_100


# In[34]:


taxi_idxs[taxi_ids_selected]


# In[27]:


pairs_errors = []
i = 0
for pair in pairs_new:
    A = classification_without_exp_random_Q(data[pair[0]], data[pair[1]], Q_size=20, 
                                            epoch=100, classifiers=[clf12])
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


# In[35]:


#print(f"test errors with Random Forest: \n {list(np.array(pairs_errors, dtype='object')[:,1])}")
test_errors_RF = [0.18, 0.29, 0.25, 0.2, 0.14, 0.07, 0.11, 0.04, 0.19, 0.24, 0.24, 0.26, 
                  0.23, 0.2, 0.17, 0.25, 0.07, 0.23, 0.18, 0.11, 0.32, 0.19, 0.29, 0.17, 
                  0.16, 0.23, 0.11, 0.18, 0.17, 0.36, 0.17, 0.2, 0.33, 0.37, 0.26, 0.21, 
                  0.24, 0.35, 0.14, 0.26, 0.14, 0.24, 0.21, 0.16, 0.22, 0.26, 0.2, 0.24, 
                  0.1, 0.21, 0.31, 0.13, 0.11, 0.33, 0.07, 0.12, 0.2, 0.32, 0.14, 0.17, 
                  0.19, 0.27, 0.27, 0.06, 0.23, 0.14, 0.22, 0.24, 0.16, 0.3, 0.18, 0.28, 
                  0.15, 0.11, 0.28, 0.26, 0.07, 0.31, 0.18, 0.2, 0.26, 0.23, 0.25, 0.3, 
                  0.13, 0.27, 0.25, 0.16, 0.14, 0.08, 0.19, 0.11, 0.22, 0.19, 0.07, 0.19, 
                  0.33, 0.19, 0.23, 0.24]
print(test_errors_RF)


# ### The following two boxes are for above 0.20 test error that newly are done

# In[36]:


pairs_100_ = pairs_100 + 0


# In[37]:


#pairs_20_ = pairs_100[np.where(np.array(pairs_errors, dtype='object')[:,1] > 0.20)[0]]
#pairs_20 = np.array(pairs_new)[np.where(np.array(pairs_errors, dtype='object')[:,1] > 0.20)[0]]

pairs_20_ = [[61, 140], [129, 153], [81, 105], [110, 127], [95, 118], [42, 93], [39, 105],
             [69, 75], [59, 80], [1, 63], [60, 113], [1, 55], [48, 92], [50, 94], [22, 51],
             [43, 53], [39, 90], [2, 40], [12, 148], [13, 45], [12, 82],  [80, 91], 
             [16, 19], [37, 73], [31, 134], [42, 145], [38, 49], [89, 106], [119, 137], 
             [30, 77], [16, 104], [66, 131], [84, 121], [83, 144], [3, 157], [70, 121], 
             [39, 146], [41, 67], [98, 138], [112, 114], [27, 149], [136, 155], [56, 68], 
             [14, 121], [8, 141], [25, 80], [7, 101], [97, 142]]

pairs_20 = [[ 61, 140], [129, 153], [ 81, 105], [110, 127], [ 95, 118], [ 42,  93], 
            [ 39, 105], [ 69,  75], [ 59,  80], [  1,  63], [ 60, 113], [  1,  55], 
            [ 48,  92], [ 50,  94], [ 22,  51], [ 43,  53], [ 39,  90], [  2,  40], 
            [ 12, 148], [ 13,  45], [ 12,  82], [ 80,  91], [ 16,  19], [ 37,  73], 
            [ 31, 134], [ 42, 145], [ 38,  49], [ 89, 106], [119, 137], [ 30,  77], 
            [ 16, 104], [ 66, 131], [ 84, 121], [ 83, 144], [  3, 157], [ 70, 121], 
            [ 39, 146], [ 41,  67], [ 98, 138], [112, 114], [ 27, 149], [136, 155], 
            [ 56,  68], [ 14, 121], [  8, 141], [ 25,  80], [  7, 101], [ 97, 142]]


print(len(pairs_20))
print(pairs_20_)
print(pairs_20)


# In[74]:


pairs_20_string = ['[ 61 140]', '[129 153]', '[ 81 105]', '[110 127]', '[ 95 118]', 
                   '[ 42  93]', '[ 39 105]', '[ 69  75]', '[ 59  80]', '[  1  63]', 
                   '[ 60 113]', '[  1  55]', '[ 48  92]', '[ 50  94]', '[ 22  51]', 
                   '[ 43  53]', '[ 39  90]', '[  2  40]', '[ 12 148]', '[ 13  45]', 
                   '[ 12  82]', '[ 80  91]', '[ 16  19]', '[ 37  73]', '[ 31 134]', 
                   '[ 42 145]', '[ 38  49]', '[ 89 106]', '[119 137]', '[ 30  77]', 
                   '[ 16 104]', '[ 66 131]', '[ 84 121]', '[ 83 144]', '[  3 157]', 
                   '[ 70 121]', '[ 39 146]', '[ 41  67]', '[ 98 138]', '[112 114]',
                   '[ 27 149]', '[136 155]', '[ 56  68]', '[ 14 121]', '[  8 141]', 
                   '[ 25  80]', '[  7 101]', '[ 97 142]']


# In[38]:


pairs_100 = np.array(pairs_new, dtype='object') + 0


# In[ ]:


#TP = pairs_in_users_in_data
pairs_in_users_in_data = [[T[pairs_20[i][0]], T[pairs_20[i][1]]] for i in range(len(pairs_20))]
print("Chosen pairs from data: \n", pairs_in_users_in_data)
pairs_in_users_in_data = np.array(pairs_in_users_in_data)
print(len(pairs_in_users_in_data))

#pairs_in_users_in_data = [[3953, 9437], [8832, 10168], [5920, 7198], [7466, 8735], 
#[6875, 8062], [2861, 6789], [2566, 7198], [4579, 4874], [3938, 5895], [250, 3967], 
#[3952, 7652], [250, 3771], [2937, 6651], [3141, 6833], [1829, 3153], [2875, 3259], 
#[2566, 6610], [816, 2593], [1349, 9836], [1377, 2908], [1349, 5969], [5895, 6622], 
#[1556, 1569], [2549, 4829], [2192, 9289], [2861, 9524], [2565, 2942], [6568, 7346], 
#[8259, 9355], [2156, 4968], [1556, 7192], [4414, 9240], [6273, 8366], [6167, 9512], 
#[819, 10200], [4581, 8366], [2566, 9787], [2616, 4451], [6969, 9367], [7623, 7673], 
#[2127, 9945], [9353, 10179], [3865, 4526], [1438, 8366], [918, 9467], [1949, 5895], 
#[876, 7015], [6907, 9480]]


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


# In[ ]:


# the following path is for above 0.20 test errors
paths = []
for i in range(len(pairs_20)):
    paths.append('Calculated Distance Matrices for KNN Above 0.20/d_Q_pi/Taxi-Pairs'+str(pairs_20[i])+'-d_Q_pi.csv')

for i in range(len(pairs_20)):
    calculate_dists_d_Q_pi(data[pairs_20[i][0]], data[pairs_20[i][1]], p=1, path=paths[i])


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


# In[60]:


# the following path is for above 0.20 test errors
train_test_median_std_errors_d_Q_pi_20 = []
paths = []
for i in range(len(pairs_20)):
    paths.append('Calculated Distance Matrices for KNN Above 0.20/d_Q_pi/Taxi-Pairs['+str(pairs_20[i])+']-d_Q_pi.csv')

for i in range(len(pairs_20)):
    print(colored(f"pair: {pairs_20[i]}", 'magenta'))
    A = KNN_average_error_d_Q_pi(data[pairs_20[i][0]], data[pairs_20[i][1]], 
                                   num_trials=1000, path_to_dists=paths[i], 
                                   pair=pairs_20[i])
    train_test_median_std_errors_d_Q_pi_20.append(A[1:])
    print(A[0])
    print("====================================================================")


# In[61]:


path = 'Calculated KNN errors for comparison with RF 20/Taxi-Pairs-d_Q_pi.csv'
np.savetxt(path, np.array(train_test_median_std_errors_d_Q_pi_20), delimiter=',')


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
    


# In[63]:


# the following path is for above 0.20 test errors
paths = []
for i in range(len(pairs_20)):
    paths.append('Calculated Distance Matrices for KNN Above 0.20/dtw-tslearn/Taxi-Pairs['+str(pairs_20[i])+']-dtw-tslearn.csv')

for i in range(len(pairs_20)):
    calculate_dists_dtw_tslearn(data[pairs_20[i][0]], data[pairs_20[i][1]], path=paths[i])


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


# In[722]:


# test errors = np.array(train_test_median_std_errors_dtw_tslearn)[:,1]


# In[66]:


# the following path is for above 0.20 test errors
paths = []
train_test_median_std_errors_dtw_tslearn_20 = []
for i in range(len(pairs_20)):
    paths.append('Calculated Distance Matrices for KNN Above 0.20/dtw-tslearn/Taxi-Pairs['+str(pairs_20[i])+']-dtw-tslearn.csv')

for i in range(len(pairs_20)):
    print(colored(f"i=, {i}, pair: {pairs_20[i]}", 'magenta'))
    A = KNN_average_error_dtw_tslearn(data[pairs_20[i][0]], data[pairs_20[i][1]], 
                                      num_trials=1000, path_to_dists=paths[i], 
                                      pair=pairs_20[i])
    train_test_median_std_errors_dtw_tslearn_20.append(A[1:])
    print(A[0])
    print("=============================================================================")


# In[67]:


path = 'Calculated KNN errors for comparison with RF 20/Taxi-Pairs-dtw-tslearn.csv'
np.savetxt(path, np.array(train_test_median_std_errors_dtw_tslearn_20), delimiter=',')


# ## From github page: https://github.com/bguillouet/traj-dist
# 
# It includes 9 distances for trajectories including: Continuous Frechet, Discrete Frechet, Hausdorff, DTW, SSPD, LCSS, EDR, ERP.
# 
# ## All but the continuous Frechet distance are really fast.

# In[301]:


import traj_dist.distance as tdist
import pickle

def calculate_dists(data, metrics, pair, eps_edr, eps_lcss): 
    start_time = time.time() 
    Data = np.concatenate((data[pair[0]], data[pair[1]]), 0)
    n = len(Data)
    for metric in metrics:
        if metric == 'edr':
            A = tdist.pdist(Data, metric, eps=eps_edr)
        elif metric == 'lcss':
            A = tdist.pdist(Data, metric, eps=eps_lcss)
        else:
            A = tdist.pdist(Data, metric)
        tri = np.zeros((n, n))
        tri[np.triu_indices(n, 1)] = A
        for i in range(1, n):
            for j in range(i):
                tri[i][j] = tri[j][i]
        # the following path is for above 0.37 test errors
        #path = 'Calculated Distance Matrices for KNN Above 0.20/7-dists/Taxi-Pairs'+str(pair)+'-'+metric+'.csv'
        #the following path is for 100 chosen pairs for correlation and R^2 test
        path = 'Calculated Distance Matrices for KNN for comparison with RF/Taxi/7-dists/Taxi-Pairs'+str(pair)+'-'+metric+'.csv'
        np.savetxt(path, tri, delimiter=',')
    total_time = time.time() - start_time
    return total_time


# In[69]:


# For pairs with above 0.20 test errors
Metrics = ['discret_frechet', 'hausdorff', 'dtw', 'sspd', 'lcss', 'edr', 'erp']

#tdist.pdist(data[pairs_20[5][1]][:20], 'edr', eps=1e-10) 
eps_edr_list = [0.005] * len(pairs_20)

#tdist.pdist(data[pairs_20[5][1]][:20], 'lcss', eps=1e-1) 
eps_lcss_list = [0.005] * len(pairs_20)

for i in range(len(pairs_20)):
    st_time = time.time()
    calculate_dists(data, metrics=Metrics, pair=pairs_20[i], eps_edr=eps_edr_list[i], 
                    eps_lcss=eps_lcss_list[i])
    print(f"time for {i}-th pair: {time.time() - st_time}")


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


# In[74]:


# the following path is for above 0.20 test errors

Metrics = ['discret_frechet', 'hausdorff', 'dtw', 'sspd', 'lcss', 'edr', 'erp']

paths = []
train_test_median_std_errors_7_dists_20 = []
for i in range(len(pairs_20)):
    paths_temp = []
    for j in range(len(Metrics)):
        paths_temp.append('Calculated Distance Matrices for KNN Above 0.20/7-dists/Taxi-Pairs'+str(pairs_20[i])+'-'+Metrics[j]+'.csv')
    paths.append(paths_temp)
    
for i in range(len(pairs_20)):
    print(colored(f"i = {i}, pair: {pairs_20[i]}", 'magenta'))
    A = KNN_average_error_7_dists(data[pairs_20[i][0]], data[pairs_20[i][1]], 
                                  num_trials=1000, dists_names=Metrics, 
                                  paths_to_dists=paths[i])
    train_test_median_std_errors_7_dists_20.append(A[3:])
    print(A[0])
    print("============================================================")


# In[75]:


for i in range(len(Metrics)):
    path = 'Calculated KNN errors for comparison with RF 20/Taxi-Pairs'+'-'+Metrics[i]+'.csv'
    np.savetxt(path, np.array(train_test_median_std_errors_7_dists_20)[:,:,i], delimiter=',')


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
    


# In[78]:


# the following path is for above 0.20 test errors
paths = []
for i in range(len(pairs_20)):
    paths.append('Calculated Distance Matrices for KNN Above 0.20/fastdtw/Taxi-Pairs['+str(pairs_20[i])+']-fastdtw.csv')

for i in range(len(pairs_20)):
    calculate_fastdtw_dists(data[pairs_20[i][0]], data[pairs_20[i][1]], path=paths[i])


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


# In[862]:


# test errors = np.array(train_test_median_std_errors_fastdtw)[:,1]


# In[81]:


# the following path is for above 0.20 test errors
paths = []
train_test_median_std_errors_fastdtw_20 = []
for i in range(len(pairs_20)):
    paths.append('Calculated Distance Matrices for KNN Above 0.20/fastdtw/Taxi-Pairs['+str(pairs_20[i])+']-fastdtw.csv')

for i in range(len(pairs_20)):
    print(colored(f"i = {i}, pair: {pairs_20[i]}", 'magenta'))
    fastdtw_matrix = np.array(pd.read_csv(paths[i], header=None))
    A = KNN_fastdtw_average_error(data[pairs_20[i][0]], data[pairs_20[i][1]], 
                                  num_trials=1000, path_to_dists= paths[i], 
                                  pair=pairs_20[i])
    train_test_median_std_errors_fastdtw_20.append(A[3:])
    print(A[0])
    print("==================================================================")


# In[82]:


path = 'Calculated KNN errors for comparison with RF 20/Taxi-Pairs-fastdtw.csv'
np.savetxt(path, np.array(train_test_median_std_errors_fastdtw_20), delimiter=',')


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
    


# In[84]:


# the following path is for above 0.20 test errors
start_time = time.time()
paths = []
for i in range(len(pairs_20)):
    paths.append('Calculated Distance Matrices for KNN Above 0.20/soft-dtw/Taxi-Pairs['+str(pairs_20[i])+']-soft-dtw.csv')

for i in range(len(pairs_20)):
    calculate_dists_soft_dtw(data[pairs_20[i][0]], data[pairs_20[i][1]], 
                             gamma=1e-14, path=paths[i])
print("total time:", time.time() - start_time)


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


# In[87]:


# test errors = np.array(train_test_median_std_errors_soft_dtw)[:,1]


# In[53]:


# the following path is for above 0.20 test errors
paths = []
train_test_median_std_errors_soft_dtw_20 = []
for i in range(len(pairs_20)):
    paths.append('Calculated Distance Matrices for KNN Above 0.20/soft-dtw/Taxi-Pairs['+str(pairs_20[i])+']-soft-dtw.csv')

print(colored(f"gamma={1e-10}", 'yellow'))
for i in range(len(pairs_20)):
    print(colored(f"i = {i}, pair: {pairs_20[i]}", 'magenta'))
    A = KNN_average_error_soft_dtw(data[pairs_20[i][0]], data[pairs_20[i][1]], 
                                     num_trials=1000, path_to_dists=paths[i], 
                                     pair=pairs_20[i])
    train_test_median_std_errors_soft_dtw_20.append(A[1:])
    print(A[0])
    print("=======================================================================")


# In[54]:


path = 'Calculated KNN errors for comparison with RF 20/Taxi-Pairs-soft_dtw.csv'
np.savetxt(path, np.array(train_test_median_std_errors_soft_dtw_20), delimiter=',')


# ## Choosing 3 pairs with best KNN test errors above 0.3

# In[24]:


test_errors = np.zeros((48,12))

Metrics_all = ['discret_frechet', 'hausdorff', 'dtw', 'sspd', 'lcss', 'edr', 'erp',
               'd_Q_pi', 'dtw-tslearn', 'fastdtw', 'soft_dtw']

for i in range(7):
    path = 'Calculated KNN errors for comparison with RF 20/Taxi-Pairs'+'-'+Metrics_all[i]+'.csv'
    test_errors[:,i] = np.array(pd.read_csv(path, header=None))[:,1]
for i in range(7, len(Metrics_all)):
    path = 'Calculated KNN errors for comparison with RF 20/Taxi-Pairs'+'-'+Metrics_all[i]+'.csv'
    test_errors[:,i] = np.array(pd.read_csv(path, header=None))[:,1]

test_errors[:,11] = np.min(test_errors[:,:-1], 1)

J = np.where(test_errors[:,-1] > 0.30)[0]
print(len(J))
print(J)
test_errors = np.round(test_errors, decimals = 4)[J]

df = pd.DataFrame(data=test_errors, index=J,
                  columns= Metrics_all + ['best error'])
df


# In[28]:


pairs_20 = np.array(pairs_20)


# In[230]:


# Do not run again
pairs_20[J][0], pairs_20[J][1], pairs_20[J][2]


# In[234]:


#taxi_idxs[taxi_ids_selected] 
print("3 of final chosen pairs")
pairs_in_users_in_data[J] + 1


# ## Choosing 2 pairs with best KNN error between 0.1 and 0.2

# In[75]:


test_errors = np.zeros((48,12))

Metrics_all = ['discret_frechet', 'hausdorff', 'dtw', 'sspd', 'lcss', 'edr', 'erp',
               'd_Q_pi', 'dtw-tslearn', 'fastdtw', 'soft_dtw']

for i in range(7):
    path = 'Calculated KNN errors for comparison with RF 20/Taxi-Pairs'+'-'+Metrics_all[i]+'.csv'
    test_errors[:,i] = np.array(pd.read_csv(path, header=None))[:,1]
for i in range(7, len(Metrics_all)):
    path = 'Calculated KNN errors for comparison with RF 20/Taxi-Pairs'+'-'+Metrics_all[i]+'.csv'
    test_errors[:,i] = np.array(pd.read_csv(path, header=None))[:,1]

test_errors[:,11] = np.min(test_errors[:,:-1], 1)

J = np.where((test_errors[:,-1] > 0.10) & (test_errors[:,-1] < 0.20))[0]
print(len(J))
print(J)
test_errors = np.round(test_errors, decimals = 4)[J]

df = pd.DataFrame(data=test_errors, index=J,
                  columns=Metrics_all + ['best error'])
df.loc[[15, 20]]


# In[59]:


test_ers = []
stds = []

for i in range(len(Metrics_all)):
    path = 'Calculated KNN errors for comparison with RF 20/Taxi-Pairs'+'-'+Metrics_all[i]+'.csv'
    array = np.array(pd.read_csv(path, header=None))
    ts_er = np.round(np.mean(array[:, 1][[13, 33, 45, 15, 20]]), decimals=4)
    test_ers.append(ts_er)
    std = np.round(np.mean(array[:, -1][[13, 33, 45, 15, 20]]), decimals=4)
    stds.append(std)


# In[68]:


print(Metrics_all)
print(test_ers)
print(stds)


# In[243]:


pairs_20[J][4], pairs_20[J][9]


# In[250]:


print("the remained 2 of final chosen pairs")
pairs_in_users_in_data[J[4]] + 1, pairs_in_users_in_data[J[9]] + 1,


# # Chosen 5 pairs

# In[190]:


pairs_final = [[50, 94], [83, 144], [25, 80], [43, 53], [12, 82]]


# In[ ]:




