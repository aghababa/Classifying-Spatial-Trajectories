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
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm
import matplotlib.pyplot as plt
from termcolor import colored
from matplotlib import colors
import pickle
from datetime import datetime
import os
from sklearn.svm import LinearSVC
from hmmlearn import hmm
from pprint import pprint 
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


# ## Do not run the following blocks until it is mentioned at very bottom

# In[2]:


N = 182 # number of users
J = glob.glob('Geolife Trajectories 1.3/**/', recursive=True)[2:]
K = [J[2*i] for i in range(N)]
F = [K[i][30:33] for i in range(N)]
int1 = np.vectorize(int)
folder_numbers = int1(F)
I = glob.glob('Geolife Trajectories 1.3/**/*.plt', recursive=True)
#np.sort(folder_numbers), len(folder_numbers)


# In[3]:


def read_file(file_name):
    data = []
    c = 0
    with open(file_name, "r") as f:
        for line in f:
            c = c + 1
            if c > 6:
                item = line.strip().split(",")
                if len(item) == 7:
                    data.append([float(item[0]), float(item[1]), float(item[4]), None])
    return np.array(data)


# In[4]:


# Test
path = glob.glob('Geolife Trajectories 1.3/Data/010/Trajectory/*.plt', recursive=True)
print(len(path)) 
print(read_file(path[5]))


# # Reading Labels

# In[5]:


I_text = glob.glob('Geolife Trajectories 1.3/**/*.txt', recursive=True)
len(I_text)


# In[6]:


labeled_users = [I_text[i][30:33] for i in range(len(I_text))]
print(labeled_users) 
print(len(labeled_users))


# In[7]:


# create 'daysDate' function to convert start and end time to a number of days

def days_date(time_str):
    date_format = "%Y/%m/%d %H:%M:%S"
    current = datetime.strptime(time_str, date_format)
    date_format = "%Y/%m/%d"
    bench = datetime.strptime('1899/12/30', date_format)
    no_days = current - bench
    delta_time_days = no_days.days + current.hour / 24.0 + current.minute / (24. * 60.) + current.second / (24. * 3600.)
    return delta_time_days


# In[8]:


# Change Mode Name to Mode index
Mode_Index = {"walk": 0, "bike": 1, "bus": 2, "car": 3, "taxi": 3, "subway": 4, 
              "railway": 4, "train": 4, "motorcycle": 5, "run": 5, "boat": 5, 
              "airplane": 5, "other": 5}

# Modes are the modes that we use here
Modes = ['walk', 'bike', 'bus', 'driving', 'train', 'others']

# We will remove 'others'


# In[9]:


def get_labels(path):
    
    labels_data = []
    c = 0
    with open(path, "r") as f:
        for line in f:
            if c > 0:
                item = line.strip().split("\t")
                if len(item) == 3:
                    labels_data.append([float(days_date(item[0])), 
                                        float(days_date(item[1])), Mode_Index[item[2]]])
            else:
                c += 1
    labels_data = np.array(labels_data)

    return set(labels_data[:,-1]), labels_data


# In[10]:


# Test
p = glob.glob('Geolife Trajectories 1.3/Data/010/labels.txt')
get_labels(p[0])


# ### Correcting date and time of label dataset according to those of trajectories

# In[11]:


n = len(I_text)
users_labels = [0] * n
E = []
for i in range(n):
    users_labels[i] = get_labels(I_text[i])[1]
    E.append(I_text[i][30:33])
users_labels = np.array(users_labels)
#users_labels[0]


# ### Determining all transportation modes in data

# In[12]:


S = set()
for path in I_text:
    S = S.union(get_labels(path)[0])
print(S)


# In[13]:


Labels = [0,1,2,3,4,5, None]
len(Labels)


# # Adding transportation mode to trajectories

# In[14]:


# Test
V = glob.glob('Geolife Trajectories 1.3/Data/010/Trajectory/*.plt', recursive = True)
pp = 'Geolife Trajectories 1.3/Data/010/labels.txt'
#np.where(get_labels(pp)[1][:,2] != None)[0]


# In[15]:


print("one line from labels data:", get_labels(pp)[1][0])
print("one line from a trajectory:", read_file(V[0])[0])


# In[16]:


def add_labels(trajectory, labels):
    count = 0
    epn = 2/(24 * 3600)
    for i in range(len(labels)):
        a1 = np.where((trajectory[:,2] >= labels[i][0] - epn))[0]
        b1 = np.where((trajectory[:,2] <= labels[i][0] + epn))[0]
        c1 = list(set(a1).intersection(b1))
        
        a2 = np.where((trajectory[:,2] >= labels[i][1] - epn))[0]
        b2 = np.where((trajectory[:,2] <= labels[i][1] + epn))[0]
        c2 = list(set(a2).intersection(b2))
        
        if len(c1) > 0 and len(c2) > 0:
            trajectory[:,3][c1[0]:c2[-1]+1] = labels[i][-1]
            count += 1
    return trajectory, count


# In[17]:


# Test
V = glob.glob('053/Trajectory/*.plt', recursive = True)
pp = '053/labels.txt'
add_labels(read_file(V[0]), get_labels(pp)[1])[0][:,-1]


# In[18]:


def add_label_to_user(user_path, labels):
    c = 0
    user = []
    for i in range(len(user_path)):
        A = add_labels(read_file(user_path[i]), labels)
        user.append(A[0])
        c += A[1]
        
    return np.array(user), c


# In[23]:


start_time = time.time()
c1 = 0 
c2 = 0
data = [0] * n
for i in range(n):
    print(i)
    user_path = glob.glob('Geolife Trajectories 1.3/Data/'+E[i]+'/trajectory/*.plt', 
                          recursive=True)
    a = add_label_to_user(user_path, users_labels[i])
    data[i] = np.array([a[1], len(users_labels[i]), a[0]])
    print('a[1], users_labels[i] =', a[1], len(users_labels[i]))
    c1 += a[1]
    c2 += len(users_labels[i])
    
data = np.array(data)
print("c1, c2 =", c1, c2)
print(time.time() - start_time)


# In[29]:


np.where(data[:,0] == 0)[0]


# In[34]:


idx = np.where(data[:,0] != 0)[0]
data = data[idx]
len(data)


# # Removing unlabeled data

# In[92]:


Data = [0] * len(data)
for i in range(len(data)):
    Data[i] = []
    for j in range(len(data[i][2])):
        I = np.where((data[i][2][j][:,3] != None) & (data[i][2][j][:,3] != 5) )[0]
        if len(I) > 0:
            Data[i].append(data[i][2][j][I])
    Data[i] = np.array(Data[i])
Data = np.array(Data)


# In[128]:


print("Total number of trajectories in the cleaned data:", 
      sum([len(Data[i]) for i in range(len(Data))]))


# # Checking labels of cleaned data

# In[129]:


Labels_final1 = []
Labels_final = set([])
for i in range(len(Data)):
    Labels_final1.append([set(Data[i][j][:,3]) for j in range(len(Data[i]))])
    for j in range(len(Data[i])):
        Labels_final = Labels_final.union(Labels_final1[i][j])
Labels_final


# # Creating csv files

# In[208]:


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


# In[199]:


DF1 = pd.DataFrame(Data[0][0]) 
DF1.to_csv(r'labeled csv Geolife/data['+str(1)+'].csv', index=False)


# In[218]:


for i in range(len(Data)):
    createFolder('labeled csv Geolife/'+str(i))
    for j in range(len(Data[i])):
        DF = pd.DataFrame(Data[i][j]) 
        DF.to_csv(r'labeled csv Geolife/'+str(i)+'/'+str(i)+'_'+str(j)+'.csv', index=False)

