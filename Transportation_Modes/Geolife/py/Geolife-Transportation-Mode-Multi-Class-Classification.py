#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob 
import time
import math
import random
import numpy as np
import pandas as pd
from scipy import linalg as LA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from termcolor import colored
from matplotlib import colors
import pickle
import os


# ## Reading cleaned csv files

# ### User i is data_1$[i]$ in the following

# In[3]:


data_1 = [0] * 57
for i in range(len(data_1)):
    fnames = glob.glob('labeled csv Geolife/'+str(i)+'/*.csv')
    data_1[i] = np.array([np.loadtxt(f, delimiter=',')[1:] for f in fnames])
data_1 = np.array(data_1)


# In[4]:


fnames = glob.glob('labeled csv Geolife/**/*.csv')
len(fnames)


# ### Users are stacked together in data_2 below

# In[5]:


data_2 = []
fnames = glob.glob('labeled csv Geolife/**/*.csv')
for f in fnames:
    data_2.append(np.loadtxt(f, delimiter=',')[1:])
data_2 = np.array(data_2)


# ### Removing segments with length less than 1e-10 because of numerical precision

# In[6]:


data_3 = [0] * len(data_2)
h = 1e-12
c = 0
for i in range(len(data_2)):
    p1 = data_2[i][:-1]
    p2 = data_2[i][1:]
    L = ((p2[:,:2]-p1[:,:2])*(p2[:,:2]-p1[:,:2])).sum(axis =1)
    I = np.where(L > h)[0]
    J = np.where(L < h)[0]
    if len(J) > 0:
        c += 1
    p1 = p1[I]
    p2 = p2[I]
    if len(I) == 0:
        print(i)
    gamma = np.concatenate((p1, p2[-1].reshape(1,4)), 0) 
    if len(gamma) > 0:
        data_3[i] = gamma
    data_3[i] = np.array(data_3[i])
data_3 = np.array(data_3)
c


# # Partitioning trajectories to less than 20 minutes long

# In[7]:


def partition(trajectory, threshold=20):
    trajectories = []
    # ("24 * 60 *" makes days_date to minutes)
    Time = 24 * 60 * (trajectory[:,2][1:] - trajectory[:,2][:-1]) # : x minutes 
    J = np.where(Time > threshold)[0]
    J = J.tolist()
    if len(J) == 0:
        trajectories.append(trajectory)
    else:
        J = [0] + J + [len(trajectory)]
        for j in range(len(J) - 1):
            trajectories.append(trajectory[J[j]:J[j+1]])
    return trajectories


# ### data_4 below is the array of trajectories having less than 20 minutes long

# In[8]:


data_4 = []
for i in range(len(data_3)):
    A = partition(data_3[i], threshold=20)
    for j in range(len(A)):
        data_4.append(A[j])
data_4 = np.array(data_4, dtype='object')
len(data_4)


# In[9]:


I = np.where(np.array([len(data_4[i]) for i in range(len(data_4))]) >= 10)[0]
data_4 = data_4[I]
len(data_4)


# In[10]:


int1 = np.vectorize(int)
data_5 = []
c = 0
for i in range(len(data_4)):
    if len(set(int1(data_4[i][:,3]))) < 2: 
        data_5.append(data_4[i])
        c += 1
data_5 = np.array(data_5)
c


# In[11]:


Modes = ['walk', 'bike', 'bus', 'driving', 'train']


# # Separating transportation modes for each trip

# In[12]:


data = []
c = np.zeros(5)

Len = np.array([len(set(int1(data_4[i][:,3]))) for i in range(len(data_4))])

K = np.where(Len == 1)[0]
c[0] = len(K)
for k in K:
    data.append(data_4[k])

for k in range(2,3):
    K = np.where(Len == k)[0]
    c[k] = len(K)
    data_temp = data_4[K]
    for i in range(len(data_temp)):
        S = set(data_temp[i][:,3])
        for s in S:
            I = np.where(data_temp[i][:,3] == s)[0]
            if (max(I) - min(I) + 1 == len(I)):
                data.append(data_temp[i][I])
            else:
                J = np.where(np.array([I[j+1]-I[j] for j in range(len(I)-1)]) != 1)[0]
                if J[0] >= 10:
                    data.append(data_temp[i][J[:J[0]]])
                for j in range(len(J)-1):
                    if (J[j+1]-J[j]) >= 10:
                        data.append(data_temp[i][J[J[j]:J[j+1]]])

data = np.array(data)
print(len(data))


# In[13]:


Lengths = np.array([len(data[i]) for i in range(len(data))])
idx = np.where(Lengths >= 10)[0]
data = data[idx]


# In[14]:


Labels_sizes = np.array([list(set(data[i][:,3]))[0] for i in range(len(data))])
for k in range(5):
    print("The number of trajectries with label", k, 'is:', len(np.where(Labels_sizes==k)[0]))


# # Multi class classification with different featuerizations

# In[15]:


import trjtrypy as tt
from trjtrypy.distances import d_Q
from trjtrypy.distances import d_Q_pi
import trjtrypy.visualizations as vs
from scipy.spatial import distance
from trjtrypy.featureMappings import curve2vec


def ExpCurve2Vec(points,curves,mu):
    D = tt.distsbase.DistsBase()
    a = np.array([np.exp(-1*np.power(D.APntSetDistACrv(points,curve),2)/(mu)**2) for curve in curves])
    return a


# ## Features

# In[16]:


# length calculator (for trajectories of one transportation mode can be used)
def length(x):
    p1 = x[:,:2][:-1]
    p2 = x[:,:2][1:]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1))
    Length = [sum(L)/len(L)]
    return Length


# In[17]:


# speed calculator (for trajectories of one transportation mode can be used)
def speed(x):
    t = x[:,2][1:] - x[:,2][:-1] + 1e-10
    p1 = x[:,:2][:-1]
    p2 = x[:,:2][1:]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1))
    speeds = L/t
    s = [sum(speeds)/len(speeds)]
    return s


# In[18]:


# acceleration calculator (for trajectories of one transportation mode can be used)
def acceleration(x):
    t = x[:,2][1:] - x[:,2][:-1] + 1e-10
    p1 = x[:,:2][:-1]
    p2 = x[:,:2][1:]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1))
    accelerations = L/t**2
    a = [1e-5 * sum(accelerations)/len(accelerations)]
    return a


# In[19]:


# jerk calculator (for trajectories of one transportation mode can be used)
def jerk(x):
    t = x[:,2][1:] - x[:,2][:-1] + 1e-10
    p1 = x[:,:2][:-1]
    p2 = x[:,:2][1:]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1))
    jerks = L/t**3
    a = [1e-5 * sum(jerks)/len(jerks)]
    return a


# # Classifiers

# In[97]:


clf1 = [make_pipeline(StandardScaler(), LinearSVC(dual=False, C=1e10, tol=1e-5, 
                               class_weight='balanced', max_iter=1000)), 
        "Linear SVM, C="+str(1e10)]
clf2 = [make_pipeline(StandardScaler(), svm.SVC(C=1000, kernel='rbf', gamma=1000, max_iter=200000)),
        "Gaussian SVM, C="+str(1000)+", gamma ="+str(1000)]
clf3 = [make_pipeline(StandardScaler(), svm.SVC(C=1000, kernel='poly', degree=3, max_iter=400000)),
        "Poly kernel SVM, C="+str(1000)+", deg=3"]
clf4 = [DecisionTreeClassifier(), "Decision Tree"]
clf5 = [RandomForestClassifier(n_estimators=100), "RandomForestClassifier, n="+str(100)]
clf6 = [KNeighborsClassifier(n_neighbors=5), "KNN"]
clf7 = [LogisticRegression(solver='newton-cg'), "Logistic Regression"]

clf = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]
classifs = [item[0] for item in clf]
keys = [item[1] for item in clf]


# In[74]:


data_traj = np.array([data[i][:,:2] for i in range(len(data))], dtype='object')


# # Classification

# ## Classification with only Physical features

# In[75]:


def select_physical_features(data, Leng=True, spd=True, accn=True, jrk=True):
    Start_time = time.time()
    
    n = len(data)
    labels = np.array([data[i][:,3][0] for i in range(n)]).reshape(-1,1)
    data_traj = np.array([data[i][:,:2] for i in range(n)])
    length_ = np.array([length(data_traj[i]) for i in range(n)]).reshape(-1,1)
    speed_ = np.array([speed(data[i]) for i in range(n)]).reshape(-1,1)
    acceleration_ = np.array([acceleration(data[i]) for i in range(n)]).reshape(-1,1)
    jerk_ = np.array([jerk(data[i]) for i in range(n)]).reshape(-1,1)
    
    A = np.concatenate((length_, speed_, acceleration_, jerk_, labels), 1)

    I = []
    if Leng == True:
        I.append(0)
    if spd == True:
        I.append(1)
    if accn == True:
        I.append(2)
    if jrk == True:
        I.append(3)
    I.append(-1)
    
    print(colored(f"Total time for mapping row data: {time.time() - Start_time}", 'green'))
    return A[:,I]


# In[76]:


def classification_physical(data, Leng=True, spd=True, accn=True, jrk=True, 
                            num_trials=10, classifiers=clf):

    Start_time = time.time()
    models = [item[0] for item in classifiers]
    keys = [item[1] for item in classifiers]
    r = len(classifiers)
    train_error_mean = np.zeros(r)
    test_error_mean = np.zeros(r)
    test_error_std = np.zeros(r)
    train_error_list = np.zeros((r, num_trials,)) 
    test_error_list = np.zeros((r, num_trials))

    for s in range(num_trials):
        
        Data = select_physical_features(data, Leng=Leng, spd=spd, accn=accn, jrk=jerk)
        X_train, X_test, y_train, y_test = train_test_split(Data[:,:-1], Data[:,-1], 
                                                            test_size=0.2)
        I = np.arange(len(X_train))
        np.random.shuffle(I)
        train_data = X_train[I]
        train_labels = y_train[I]

        J = np.arange(len(X_test))
        np.random.shuffle(J)
        test_data = X_test[J]
        test_labels = y_test[J]
        
        for k in range(r):            
            model = models[k]

            #Train the model using the training sets
            model.fit(train_data, train_labels)

            #Predict train labels
            train_pred = model.predict(train_data)
            err = 1 - metrics.accuracy_score(train_labels, train_pred)
            train_error_list[k][s] = err
            
            #Predict test labels
            test_pred = model.predict(test_data)
            er = 1 - metrics.accuracy_score(test_labels, test_pred)
            test_error_list[k][s] = er
            
    for k in range(r):
        train_error_mean[k] = np.mean(train_error_list[k])
        test_error_mean[k] = np.mean(test_error_list[k])
        test_error_std[k] = np.std(test_error_list[k])
    
    Dic = {}
    for k in range(len(keys)): 
        Dic[k] = [keys[k], np.round(train_error_mean[k], decimals = 4), 
                    np.round(test_error_mean[k], decimals = 4),
                    np.round(test_error_std[k], decimals = 4)]

    pdf = pd.DataFrame.from_dict(Dic, orient='index', columns=['Classifier','Train Error', 
                                                               'Test Error', 'Std Error'])
    print(colored(f"Total time: {time.time() - Start_time}", 'red'))
    return pdf


# ### Classification with just physical features

# In[77]:


classification_physical(data, Leng=True, spd=True, accn=True, jrk=True, num_trials=50, 
                        classifiers=clf)


# ## Classification with generating random Q in each iteration

# In[78]:


def select_features_Q(data, version='unsigned', sigma=1, Q_size=20, Leng=False, 
                      spd=False, accn=False, jrk=False, normal=False):
    Start_time = time.time()
    
    n = len(data)
    labels = np.array([data[i][:,3][0] for i in range(n)]).reshape(-1,1)
    data_traj = np.array([data[i][:,:2] for i in range(n)])

    Min = np.min([np.min(data_traj[i], 0) for i in range(n)], 0)
    Max = np.max([np.max(data_traj[i], 0) for i in range(n)], 0)
    Mean = np.mean([np.mean(data_traj[i], 0) for i in range(n)], 0)
    Std = np.std([np.std(data_traj[i], 0) for i in range(n)], 0)
    
    Q = np.ones((Q_size,2))
    Q[:,0] = 0.5 * np.random.random_sample(Q_size) + Mean[0] + 0.1
    Q[:,1] = 0.5 * np.random.random_sample(Q_size) + Mean[1] + 0.6

    projected = np.array(curve2vec(Q, data_traj, version=version, sigma=sigma))
    length_ = np.array([length(data_traj[i]) for i in range(n)]).reshape(-1,1)
    speed_ = np.array([speed(data[i]) for i in range(n)]).reshape(-1,1)
    acceleration_ = np.array([acceleration(data[i]) for i in range(n)]).reshape(-1,1)
    jerk_ = np.array([jerk(data[i]) for i in range(n)]).reshape(-1,1)
    
    A = np.concatenate((projected, length_, speed_, acceleration_, jerk_, labels), 1)

    I = list(np.arange(Q_size))
    if Leng == True:
        I.append(Q_size)
    if spd == True:
        I.append(Q_size+1)
    if accn == True:
        I.append(Q_size+2)
    if jrk == True:
        I.append(Q_size+3)
    I.append(-1)
    
    if normal == True:
        A[:,I][:-1] = (A[:,I][:-1]-np.mean(A[:,I][:-1], 0))/(np.std(A[:,I][:-1], 0))
    
    print(colored(f"Total time for mapping row data: {time.time() - Start_time}", 'green'))
    return A[:,I]


# In[79]:


def classification_Q(data, version='unsigned', sigma=1, Q_size=20, Leng=True, spd=True,
                     accn=True, jrk=True, num_trials=10, classifiers=clf, normal=False):

    Start_time = time.time()
    models = [item[0] for item in classifiers]
    keys = [item[1] for item in classifiers]
    r = len(classifiers)
    train_error_mean = np.zeros(r)
    test_error_mean = np.zeros(r)
    test_error_std = np.zeros(r)
    train_error_list = np.zeros((r, num_trials,)) 
    test_error_list = np.zeros((r, num_trials))

    for s in range(num_trials):
        
        Data = select_features_Q(data, version, sigma=sigma, Q_size=Q_size, Leng=Leng, 
                                 spd=spd, accn=accn, jrk=jerk)

        X_train, X_test, y_train, y_test = train_test_split(Data[:,:-1], Data[:,-1], 
                                                            test_size=0.2)
        
        I = np.arange(len(X_train))
        np.random.shuffle(I)
        train_data = X_train[I]
        train_labels = y_train[I]

        J = np.arange(len(X_test))
        np.random.shuffle(J)
        test_data = X_test[J]
        test_labels = y_test[J]
        
        for k in range(r):            
            model = models[k]

            #Train the model using the training sets
            model.fit(train_data, train_labels)

            #Predict train labels
            train_pred = model.predict(train_data)
            err = 1 - metrics.accuracy_score(train_labels, train_pred)
            train_error_list[k][s] = err
            
            #Predict test labels
            test_pred = model.predict(test_data)
            er = 1 - metrics.accuracy_score(test_labels, test_pred)
            test_error_list[k][s] = er
            
    for k in range(r):
        train_error_mean[k] = np.mean(train_error_list[k])
        test_error_mean[k] = np.mean(test_error_list[k])
        test_error_std[k] = np.std(test_error_list[k])
    
    Dic = {}

    for k in range(len(keys)): 
        Dic[k] = [keys[k], np.round(train_error_mean[k], decimals = 4), 
                    np.round(test_error_mean[k], decimals = 4),
                    np.round(test_error_std[k], decimals = 4)]

    pdf = pd.DataFrame.from_dict(Dic, orient='index', columns=['Classifier','Train Error', 
                                                               'Test Error', 'Std Error'])
    print(colored(f"Total time: {time.time() - Start_time}", 'red'))
    return pdf


# ### Classification with just $v_Q$ with random Q in each iteration

# In[80]:


classification_Q(data, version='unsigned', Q_size=20, Leng=False, spd=False,
                 accn=False, jrk=False, num_trials=50, classifiers=clf, normal=False)


# ### Classification with just $v_Q^{\varsigma}$ with random Q in each iteration

# In[81]:


classification_Q(data, version='signed', sigma=1, Q_size=20, Leng=False, spd=False,
                 accn=False, jrk=False, num_trials=50, classifiers=clf, normal=False)


# ### Classification with $v_Q^+$ with random Q in each iteration

# In[88]:


classification_Q(data, version='unsigned', Q_size=20, Leng=True, spd=True,
                 accn=True, jrk=True, num_trials=50, classifiers=clf, normal=False)


# ### Classification with $v_Q^{\varsigma +}$ with random Q in each iteration

# In[83]:


classification_Q(data, version='signed', sigma=1, Q_size=20, Leng=True, spd=True,
                 accn=True, jrk=True, num_trials=50, classifiers=clf, normal=False)


# ## Classification with $v_Q^+$ with choosing Q in each iteration by the perceptron-like algorithm
# 
# This didn't work better than random choice of Q

# In[89]:


def get_mu_(data_1, data_2):
    a = np.mean([np.mean(data_1[i], 0) for i in range(len(data_1))], 0)
    b = np.mean([np.mean(data_2[i], 0) for i in range(len(data_2))], 0)
    c = abs(a-b)
    return max(c)


modes = [0, 1, 2, 3, 4]
def get_mu(train):
    mus = []
    train_traj = np.array([train[i][:,:2] for i in range(len(train))])
    for i in range(len(modes)-1):
        for j in range(i+1, len(modes)):
            I = np.where(np.array([train[k][0][-1] for k in range(len(train))]) == i)[0]
            J = np.where(np.array([train[k][0][-1] for k in range(len(train))]) == j)[0]
            mus.append(get_mu_(train_traj[I], train_traj[J]))
    return np.max(mus)


# In[90]:


from scipy.stats import entropy

def initialize_Q(train, C, gamma, Q_size, mu_coeff, model, Leng=True, spd=True, accn=True, 
                 jrk=True, n_estimators=100, n_neighbors=5, deg=11):
    Start_time = time.time()
    n = len(train)
    train_traj = np.array([train[i][:,:2] for i in range(n)])
    train_labels = np.array([list(set(train[i][:,-1]))[0] for i in range(n)])
    length_ = np.array([length(train_traj[i]) for i in range(n)]).reshape(-1,1)
    speed_ = np.array([speed(train[i]) for i in range(n)]).reshape(-1,1)
    acceleration_ = np.array([acceleration(train[i]) for i in range(n)]).reshape(-1,1)
    jerk_ = np.array([jerk(train[i]) for i in range(n)]).reshape(-1,1)
    
# Make the classifier
    if model == 'LSVM':
        Model = make_pipeline(LinearSVC(dual=False, C=C, tol=1e-5, class_weight='balanced', max_iter=1000))
    elif model == 'GSVM':
        Model = make_pipeline(StandardScaler(), svm.SVC(C=C, kernel='rbf', gamma=gamma, max_iter=200000))
    elif model == 'PSVM':
        Model = make_pipeline(StandardScaler(), svm.SVC(C=C, kernel='poly', degree=deg, max_iter=400000))
    elif model == 'DT':
        Model = DecisionTreeClassifier()
    elif model == 'RF':
        Model = RandomForestClassifier(n_estimators=n_estimators)
    elif model == 'KNN':
        Model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model == 'LR':
        Model = LogisticRegression(solver='lbfgs')
    else:
        print("model is not supported")
        
    #print(train.shape, train[0].shape)
    mu = get_mu(train) * mu_coeff
    std = mu/2
    errors = []
    Q = []
    Train_0 = np.concatenate((length_, speed_, acceleration_, jerk_), 1)
    #Train_0 = length_
    Model.fit(Train_0, train_labels)
    train_pred = Model.predict(Train_0) 
    error = 1 - metrics.accuracy_score(train_labels, train_pred)
    errors.append(error)
    
    #I = np.where(train_pred != train_labels)[0]
    I = np.arange(len(train_labels))
    if model in ['LSVM', 'GSVM', 'LR', 'PSVM']:
        J = np.max(Model.decision_function(Train_0[I]), 1)
    elif model in ['DT', 'RF', 'KNN']:
        J = np.max(entropy(Model.predict_proba(Train_0[I]), axis=1))

    index = I[np.argmax(J)]
    k = np.random.randint(0, high=len(train_traj[index]))
    q = train_traj[index][k] + np.random.normal(0, std, 2)
    Q.append(q)

# Iteratively choose landmarks
    for i in range(1, Q_size):
        Train = np.concatenate((Train_0, np.array(ExpCurve2Vec(np.array(Q), train_traj, mu))), 1)
        Model.fit(Train, train_labels)
        train_pred = Model.predict(Train)
        
        error = 1 - metrics.accuracy_score(train_labels, train_pred)
        errors.append(error)
        
        #K = np.where(train_pred != train_labels)[0] # was I previously
        I = np.arange(len(train_labels))
        #if len(K) == 0: 
        #    return np.array(Q), mu, np.array(errors), error
        if model in ['LSVM', 'GSVM', 'LR', 'PSVM']:
            J = np.max(Model.decision_function(Train[I]), 1)
        elif model in ['DT', 'RF', 'KNN']:
            J = np.max(entropy(Model.predict_proba(Train[I]), axis=1))
        
        index = I[np.argmax(J)]
        k = np.random.randint(0, high=len(train_traj[index]))
        q = train_traj[index][k] + np.random.normal(0, std, 2)
        Q.append(q)

    Train = np.concatenate((Train_0, np.array(ExpCurve2Vec(np.array(Q), train_traj, mu))), 1)
    Model.fit(Train, train_labels)
    train_pred = Model.predict(Train)

    error = 1 - metrics.accuracy_score(train_labels, train_pred)
    errors.append(error)

    print(colored(f"Total time for mapping row data: {time.time() - Start_time}", 'green'))

    return np.array(Q), mu, np.array(errors), error


# In[91]:


def classification_init_Q(data, C, gamma, Q_size, mu_coeff, model, epoch, init_iter, 
                          classifiers, Leng=True, spd=True, accn=True, jrk=True, 
                          n_estimators=100, n_neighbors=5, deg=11):

    start_time = time.time()
    models = [item[0] for item in classifiers]
    keys = [item[1] for item in classifiers]

    r = len(classifiers)

    train_error_mean = np.zeros(r)
    test_error_mean = np.zeros(r)
    test_error_std = np.zeros(r)
    
    train_errors = np.zeros((r, epoch)) 
    test_errors = np.zeros((r, epoch))

    n = len(data) 
    data_traj = np.array([data[i][:,:2] for i in range(n)])
    labels = np.array([list(set(data[i][:,-1]))[0] for i in range(n)])
    length_ = np.array([length(data_traj[i]) for i in range(n)]).reshape(-1,1)

    for s in range(epoch):
        I = np.arange(n)
        np.random.shuffle(I)
        train = data_traj[I][:int(0.8 * n)]
        test = data_traj[I][int(0.8 * n):]
        train_labels = labels[I][:int(0.8 * n)]
        test_labels = labels[I][int(0.8 * n):]
        
        x_preds = np.zeros((r, len(train)))
        y_preds = np.zeros((r, len(test)))

        Q_list = []
        temp_errors = []
        mu_list = []
        for j in range(init_iter):
            B = initialize_Q(data[I][:int(0.8 * n)], C=C, gamma=gamma, Q_size=Q_size, 
                             mu_coeff=mu_coeff, model=model, Leng=Leng, spd=spd, 
                             accn=accn, jrk=jrk, n_estimators=n_estimators, 
                             n_neighbors=n_neighbors, deg=deg)
            
            Q_list.append(B[0])
            temp_errors.append(B[-1])
            mu_list.append(B[1])

        h = np.argmin(temp_errors)
        Q = Q_list[h]
        mu = mu_list[h]
        print("mu =", mu)

        train_data = np.concatenate((length_[I][:int(0.8 * n)], 
                                     np.array(ExpCurve2Vec(Q, train, mu))), 1)
        test_data = np.concatenate((length_[I][int(0.8 * n):], 
                                    np.array(ExpCurve2Vec(Q, test, mu))), 1)

        for k in range(r): 
            Model = models[k]
            Model.fit(train_data, train_labels)
            x_preds[k] = Model.predict(train_data)                
            y_preds[k] = Model.predict(test_data)

        for k in range(r):
            train_errors[k][s] = 1 - metrics.accuracy_score(train_labels, x_preds[k])
            test_errors[k][s] = 1 - metrics.accuracy_score(test_labels, y_preds[k])
            
    for k in range(r):
        train_error_mean[k] = np.mean(train_errors[k])
        test_error_mean[k] = np.mean(test_errors[k])
        test_error_std[k] = np.std(test_errors[k])
    
    Dict = {}

    for k in range(len(keys)): 
        Dict[k+1] = [keys[k], np.round(train_error_mean[k], decimals=4), 
                     np.round(test_error_mean[k], decimals=4),
                     np.round(test_error_std[k], decimals=4)]

    pdf = pd.DataFrame.from_dict(Dict, orient='index', 
                columns=['Classifier','Train Error', 'Test Error', 'Std Error'])
    
    print(colored(f"total time = {time.time() - start_time}", "red"))

    return pdf, train_error_mean, test_error_mean


# In[93]:


L = classification_init_Q(data, C=1e10, gamma=1, Q_size=20, mu_coeff=1, model='LSVM', 
                          epoch=50, init_iter=3, classifiers=clf, Leng=True, spd=True, 
                          accn=True, jrk=True, n_estimators=100, n_neighbors=5)
L[0]


# In[54]:


G = classification_init_Q(data, C=1e3, gamma=1e2, Q_size=20, mu_coeff=1, model='GSVM', 
                          epoch=50, init_iter=3, classifiers=clf, Leng=True, spd=True, 
                          accn=True, jrk=True, n_estimators=100, n_neighbors=5)
G[0]


# In[92]:


P = classification_init_Q(data, C=1e3, gamma=1, Q_size=20, mu_coeff=1, model='PSVM', 
                          epoch=50, init_iter=3, classifiers=clf, Leng=True, spd=True, 
                          accn=True, jrk=True, n_estimators=100, n_neighbors=5, deg=11)
P[0]


# In[106]:


D = classification_init_Q(data, C=1, gamma=1, Q_size=20, mu_coeff=1, model='DT', 
                          epoch=50, init_iter=3, classifiers=clf, Leng=True, spd=True, 
                          accn=True, jrk=True, n_estimators=100, n_neighbors=5)
D[0]


# In[107]:


R = classification_init_Q(data, C=1, gamma=1, Q_size=20, mu_coeff=1, model='RF', 
                          epoch=50, init_iter=3, classifiers=clf, Leng=True, spd=True, 
                          accn=True, jrk=True, n_estimators=100, n_neighbors=5)
R[0]


# In[108]:


K = classification_init_Q(data, C=1, gamma=1, Q_size=20, mu_coeff=1, model='KNN', 
                          epoch=50, init_iter=3, classifiers=clf, Leng=True, spd=True, 
                          accn=True, jrk=True, n_estimators=100, n_neighbors=5)
K[0]


# In[42]:


LR = classification_init_Q(data, C=1, gamma=1, Q_size=20, mu_coeff=1, model='LR', 
                          epoch=50, init_iter=3, classifiers=clf, Leng=True, spd=True, 
                          accn=True, jrk=True, n_estimators=100, n_neighbors=5)
LR[0]


# In[ ]:




