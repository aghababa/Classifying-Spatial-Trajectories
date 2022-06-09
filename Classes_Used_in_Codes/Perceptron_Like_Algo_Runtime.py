import numpy as np 
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
from termcolor import colored
from sklearn.svm import LinearSVC
import trjtrypy as tt
from trjtrypy.distances import d_Q
from trjtrypy.distances import d_Q_pi
from trjtrypy.featureMappings import curve2vec
from scipy.spatial import distance
from collections import Counter
import time
from scipy.stats import entropy




def ExpCurve2Vec(points, curves, mu):
    D = tt.distsbase.DistsBase()
    a = np.array([np.exp(-1*np.power(D.APntSetDistACrv(points,curve),2)/(mu)**2) for curve in curves])
    return a


# models = ['LSVM', 'GSVM', 'PSVM', 'LR', 'PERN', 'DT', 'RF', 'KNN']

class runTimeMD:
    def __init__(self, data_1, data_2, Q_size, model, C, gamma, std_coeff, 
                 n_neighbors, n_estimators):
        
        self.data_1 = data_1
        self.data_2 = data_2
        self.Q_size = Q_size
        self.model = model
        self.C = C
        self.gamma = gamma
        self.std_coeff = std_coeff
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        
        
    
        
    '''mu calculator function'''
        
    def get_mu(self):
        a = np.mean([np.mean(self.data_1[i], 0) for i in range(len(self.data_1))], 0)
        b = np.mean([np.mean(self.data_2[i], 0) for i in range(len(self.data_2))], 0)
        return max(abs(a-b)) * self.std_coeff
       
        
    '''Perceptron-Like Algorithm'''
        
    def initialize_Q(self): 
        
        Q = []        

        mu = self.get_mu()
        std = mu * self.std_coeff

        trajectory_train_data = np.concatenate((self.data_1, self.data_2), axis = 0)
        train_labels = np.concatenate(([1] * len(self.data_1), [-1] * len(self.data_2)), 0)
        index = np.random.randint(0, high=len(trajectory_train_data)) 
        k = np.random.randint(0, high=len(trajectory_train_data[index]))
        q = trajectory_train_data[index][k] + np.random.normal(0, std, 2)
        Q.append(q)
        
        if self.model == "LSVM":
            clf = make_pipeline(LinearSVC(dual=False, C=self.C, tol=1e-5, 
                                        class_weight='balanced', max_iter=1000))
        elif self.model == "GSVM":
            clf = make_pipeline(StandardScaler(), svm.SVC(C=self.C, kernel='rbf', 
                                                gamma=self.gamma, max_iter=200000))
        elif self.model == 'PSVM':
            clf = make_pipeline(StandardScaler(), svm.SVC(C=self.C, kernel='poly', 
                                                    degree=3, max_iter = 400000))
        elif self.model == "LR":
            clf = LogisticRegression(solver='lbfgs')
        elif self.model == "Prn":
            clf = Perceptron(tol=1e-5, validation_fraction=0.01, class_weight="balanced")
        elif self.model == "DT":
            clf = DecisionTreeClassifier()
        elif self.model == "RF":
            clf = RandomForestClassifier(n_estimators=self.n_estimators)
        elif self.model == "KNN":
            clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        else:
            print("error: model is not supported")
        
        for i in range(self.Q_size):
            train_data = ExpCurve2Vec(np.array(Q), trajectory_train_data, mu)
            clf.fit(train_data, train_labels)

            train_pred = clf.predict(train_data)
            error = sum(train_labels != train_pred)/len(train_labels)
            
            if self.model in ['LSVM', 'GSVM', 'PSVM', 'LR', 'Prn']:
                temp = train_labels * clf.decision_function(train_data)
                index = np.argmin(temp)
            elif self.model in ['DT', 'RF', 'KNN']:
                probs = clf.predict_proba(train_data)
                index = np.argmax(entropy(probs, axis=1))
            
            k = np.random.randint(0, high=len(trajectory_train_data[index]))
            q = trajectory_train_data[index][k] + np.random.normal(0, std, 2)
            Q.append(q)
        
        return np.array(Q), mu



    
    def train_classifier_runtime_Q(self):

        if self.model == 'LSVM':
            clf_L = [make_pipeline(LinearSVC(dual=False, C=self.C, tol=1e-5, 
                                       class_weight ='balanced', max_iter=1000)), 
                    "SVM, Linear SVC, C="+str(self.C)]
            clf = clf_L[0] 
        elif self.model == 'GSVM':
            clf_rbf = [make_pipeline(StandardScaler(), svm.SVC(C=self.C, kernel='rbf', 
                                                    gamma=self.gamma, max_iter=200000)), 
                       "GSVM, C="+str(self.C)+", gamma="+str(self.gamma)]
            clf = clf_rbf[0]
        elif self.model == 'PSVM':
            clf_PSVM = [make_pipeline(StandardScaler(), svm.SVC(C=self.C, kernel='poly', 
                                                        degree=3, max_iter = 400000)),
                        "Poly kernel SVM, C="+str(self.C)+", deg=auto"]
            clf = clf_PSVM[0]
        elif self.model == "LR":
            clf_LR = [LogisticRegression(solver='lbfgs'), "Logistic Regression"]
            clf = clf_LR[0]
        elif self.model == "Prn":
            clf_Prn = [Perceptron(tol=1e-5, validation_fraction=0.01, 
                               class_weight="balanced"), "Perceptron"]
            clf = clf_Prn [0]
        elif self.model == "DT":
            clf_DT = [DecisionTreeClassifier(), "Decision Tree"]
            clf = clf_DT[0]
        elif self.model == "RF":
            clf_RF = [RandomForestClassifier(n_estimators=self.n_estimators), 
                             "RandomForestClassifier, n="+str(self.n_estimators)]
            clf = clf_RF[0]
        elif self.model == "KNN":
            clf_KNN = [KNeighborsClassifier(n_neighbors=self.n_neighbors), "KNN"]
            clf = clf_KNN[0]
        else:
            print('model is not supported')

        start_time = time.time()

        n_1 = len(self.data_1)
        n_2 = len(self.data_2) 
        data = np.concatenate((self.data_1, self.data_2), axis = 0)
        labels = np.array([1]*n_1 + [-1]*n_2)
        
        I = np.arange(len(data))
        np.random.shuffle(I)
        data = data[I]
        labels = labels[I]

        init_Q_strat_time = time.time()
        B = self.initialize_Q()
        init_Q_end_time = time.time()
        init_total_time = init_Q_end_time - init_Q_strat_time
        
        Q = B[0]
        mu = B[1]
        Data = ExpCurve2Vec(Q, data, mu)
        clf.fit(Data, labels)

        train_runtime = time.time() - start_time

        return train_runtime, init_total_time
