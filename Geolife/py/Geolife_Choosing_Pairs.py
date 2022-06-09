#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np 
import time
import random
from scipy import linalg as LA
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from termcolor import colored
from sklearn.svm import LinearSVC


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


# In[10]:


# 24 * 60 * (days_date('1899/12/30 2:50:06') - days_date('1899/12/30 2:20:06')) == 20 min
Time = [0] * len(data_1)
for i in range(len(data_1)):
    Time[i] = []
    for j in range(len(data_1[i])):
        Time[i].append(24 * 60 * sum(data_1[i][j][:,2][1:] - data_1[i][j][:,2][:-1])) # = 20 minutes 
    Time[i] = np.array(Time[i], dtype='object')

Time = np.array(Time, dtype='object')
Time.shape


# In[11]:


J = [np.where(Time[i] > 20)[0] for i in range(len(Time))]
print(len(J))


# In[12]:


# Check to see if partitioning into less than 20 minutes worked correctly
for j in J[0]:
    A = partition(data_1[0][j], threshold=20)
    B = np.array([24 * 60 * sum(A[i][:,2][1:] - A[i][:,2][:-1]) for i in range(len(A))], dtype='object')
    I = np.where(B > 20)[0]
    if len(I) > 0: 
        print(j)


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


# In[14]:


A = [len(data_1[0][i]) for i in range(len(data_1[0]))]
print(A)


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


# In[19]:


np.sort(list(map(len, data2)))


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

clf = [clf0, clf1, clf2, clf3, clf4, clf5, clf6]
classifs = [item[0] for item in clf]
keys = [item[1] for item in clf]


# # Choosing some users

# In[27]:


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
            train_errors[k][s] = sum(y != x_preds[k])/len(y)
            test_errors[k][s] = sum(test_labels != y_preds[k])/len(test_labels)
            
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


# In[24]:


# *********** Don't run again ***********
pairs_all = []
pairs_20 = []
user_idx_20_in_data2 = []
users_pairs_in_data2 = []
for i in range(len(data2)-1):
    for j in range(i+1, len(data2)):
        A = classification_without_exp_random_Q(data2[i], data2[j], Q_size=20, epoch=10,
                                                classifiers=[clf12])
        pairs_all.append([[i, j], np.round(A[-1][0], decimals=2)])
        if A[-1][0] >= 0.20: 
            pairs_20.append([[i, j], np.round(A[-1][0], decimals=2)])
            user_idx_20_in_data2.append(user_idx[i])
            user_idx_20_in_data2.append(user_idx[j])
            users_pairs_in_data2.append([user_idx[i], user_idx[j]])
            print(colored(f"i,j={i},{j}", 'green'))
        print(A[0])
        print("===================================================================")
user_idx_20_in_data2 = list(map(int, user_idx_20_in_data2))
print("pairs having at least 20% test error:", np.array(pairs_20, dtype='object')[:,0])
print("pairs_20:", pairs_20)
print(colored(f"users = {list(set(chosen_users[user_idx_20_in_data2]))}", 'blue'))
print("users_pairs_in_data2 =", users_pairs_in_data2)


# In[27]:


pairs_all


# In[23]:


pairs_20 = [[[4, 12], 0.2], [[4, 16], 0.24], [[5, 12], 0.2], [[8, 10], 0.27]]
pairs_20


# In[24]:


pairs = list(np.array(pairs_20, dtype='object')[:,0])
pairs


# In[26]:


print("chosen pairs from data2:", pairs)


# In[27]:


print("errors of chose users:", np.sort(np.array(pairs_20, dtype='object')[:,1]))


# In[37]:


# *********** Don't run again ***********
for i in range(len(pairs)):
    A = classification_without_exp_random_Q(data2[pairs[i][0]], 
                                            data2[pairs[i][1]], Q_size=20, 
                                            epoch=10, classifiers=clf)
    print(colored(f"i = {i}, pair = {pairs[i]}", 'green'))
    print(A[0])
    print("===================================================================")


# In[38]:


# *********** Don't run again ***********
# This is only with Random Forest
pairs = np.array(pairs_20)[:,0]
for pair in pairs:
    A = classification_without_exp_random_Q(data2[pair[0]], 
                                            data2[pair[1]], Q_size=20, 
                                            epoch=100, classifiers=[clf12])
    print(colored(f"pair={pair}", 'green'))
    print(A[0])
    print("===================================================================")


# In[29]:


# Don't run again
A = set([pair[0] for pair in list(np.array(pairs_20, dtype='object')[:,0])])
B = set([pair[1] for pair in list(np.array(pairs_20, dtype='object')[:,0])])
I = list(A.union(B))
I


# In[29]:


data = data2.copy()
pairs_final = list(np.array(pairs_20, dtype='object')[:,0]) 

# use data and pairs_final


# In[30]:


pairs_final
#pairs_final = [[4, 12], [4, 16], [5, 12], [8, 10]]

