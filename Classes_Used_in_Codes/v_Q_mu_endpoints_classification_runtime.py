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
import timeit



def ExpCurve2Vec(points, curves, mu):
    D = tt.distsbase.DistsBase()
    a = np.array([np.exp(-1*np.power(D.APntSetDistACrv(points,curve),2)/(mu)**2) for curve in curves])
    return a


'''The following class includes 3 functions:
        1. According to v_Q classification (classification_v_Q() function)
            a) Average
            b) Average-Majority
        2. According to v_Q^mu classification (classification_v_Q_mu() function)
            a) Average
            b) Average-Majority
        3. Endpoint classification (endpoint_classification() function)
            a) Only average'''

class runTime():
    def __init__(self, data_1, data_2, Q_size, classifiers, std_coeff=1, version='unsigned', sigma=1):
        self.data_1 = data_1
        self.data_2 = data_2
        self.Q_size = Q_size
        self.classifiers = classifiers
        self.version = version
        self.sigma = sigma
        self.std_coeff = std_coeff

    
        
    '''mu calculator function'''
        
    def get_mu(self, data_1, data_2):
        a = np.mean([np.mean(data_1[i], 0) for i in range(len(data_1))], 0)
        b = np.mean([np.mean(data_2[i], 0) for i in range(len(data_2))], 0)
        return max(abs(a-b)) * self.std_coeff


    def generate_Q_runtime(self):

        start_time = time.time()
        data = np.concatenate((self.data_1, self.data_2), 0)

        Mean = np.mean([np.mean(data[i], 0) for i in range(len(data))], 0)
        Std = np.std([np.std(data[i], 0) for i in range(len(data))], 0)

        Q = np.ones((self.Q_size, 2))
        Q[:,0] = np.random.normal(Mean[0], 4 * Std[0], self.Q_size)
        Q[:,1] = np.random.normal(Mean[1], 4 * Std[1], self.Q_size)

        stop_time = time.time()
        total_time = stop_time - start_time

        return total_time


    def train_classifier_v_Q_runtime(self):

        start_time = time.time()
        models = [item[0] for item in self.classifiers]
        keys = [item[1] for item in self.classifiers]

        r = len(self.classifiers)
        train_runtime = np.zeros(r)

        for k in range(r): 

            time_temp = time.time() #timeit.default_timer()

            n_1 = len(self.data_1)
            n_2 = len(self.data_2) 

            data = np.concatenate((self.data_1, self.data_2), 0)
            labels = np.array([1]*n_1 + [-1]*n_2)

            Mean = np.mean([np.mean(data[i], 0) for i in range(len(data))], 0)
            Std = np.std([np.std(data[i], 0) for i in range(len(data))], 0)
            
            Q = np.ones((self.Q_size, 2))
            Q[:,0] = np.random.normal(Mean[0], 4 * Std[0], self.Q_size)
            Q[:,1] = np.random.normal(Mean[1], 4 * Std[1], self.Q_size)

            Data = curve2vec(Q, data, version=self.version, sigma=self.sigma)

            model = models[k]
            model.fit(Data, labels)
            stop_time = time.time() #timeit.default_timer()

            train_runtime[k] = stop_time - time_temp
                    
        print(colored(f"run time: {time.time() - start_time}", "red"))
        
        Dict = {}
        for k in range(len(keys)): 
            Dict[k+1] = [keys[k], train_runtime[k]]

        pdf = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier','Runtime'])

        return pdf, train_runtime


    def train_classifier_v_Q_exp_runtime(self):

        start_time = time.time()
        models = [item[0] for item in self.classifiers]
        keys = [item[1] for item in self.classifiers]

        r = len(self.classifiers)
        train_runtime = np.zeros(r)

        for k in range(r): 

            time_temp = time.time() #timeit.default_timer()

            n_1 = len(self.data_1)
            n_2 = len(self.data_2) 

            data = np.concatenate((self.data_1, self.data_2), 0)
            labels = np.array([1]*n_1 + [-1]*n_2)

            mu = self.get_mu(self.data_1, self.data_2)

            Mean = np.mean([np.mean(data[i], 0) for i in range(len(data))], 0)
            Std = np.std([np.std(data[i], 0) for i in range(len(data))], 0)
            
            Q = np.ones((self.Q_size, 2))
            Q[:,0] = np.random.normal(Mean[0], 4 * Std[0], self.Q_size)
            Q[:,1] = np.random.normal(Mean[1], 4 * Std[1], self.Q_size)

            Data = ExpCurve2Vec(Q, data, mu)

            model = models[k]
            model.fit(Data, labels)
            stop_time = time.time() #timeit.default_timer()

            train_runtime[k] = stop_time - time_temp
                    
        print(colored(f"run time: {time.time() - start_time}", "red"))
        
        Dict = {}
        for k in range(len(keys)): 
            Dict[k+1] = [keys[k], train_runtime[k]]

        pdf = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier','Runtime'])

        return pdf, train_runtime

    '''Get the endpoints of all trajectories in data_1 and data_2'''
    def get_endpoints(self):
        n_1 = len(self.data_1)
        n_2 = len(self.data_2)
        data_1_endpoints = np.zeros((n_1, 4))
        data_2_endpoints = np.zeros((n_2, 4))
        for i in range(n_1):
            data_1_endpoints[i] = np.concatenate((self.data_1[i][0], self.data_1[i][-1]), 0)
        for i in range(n_2):
            data_2_endpoints[i] = np.concatenate((self.data_2[i][0], self.data_2[i][-1]), 0)
        return data_1_endpoints, data_2_endpoints
        

    '''Get the endpoint classification error
       parameters: 
            1. data_1, data_2
            2. epoch
            3. classifiers'''
    def train_classifier_endpoint_runtime(self):

        start_time = time.time()
        models = [item[0] for item in self.classifiers]
        keys = [item[1] for item in self.classifiers]
        r = len(self.classifiers)
        train_runtime = np.zeros(r)
        n_1 = len(self.data_1)
        n_2 = len(self.data_2) 

        data_1_endpoints, data_2_endpoints = self.get_endpoints()
        Data = np.concatenate((data_1_endpoints, data_2_endpoints), 0)
        labels = np.array([1]*n_1 + [-1]*n_2)

        for k in range(r):
            time_temp = time.time()
            model = models[k]
            model.fit(Data, labels)
            stop_time = time.time()
            train_runtime[k] = stop_time - time_temp
                    
        print(colored(f"run time: {time.time() - start_time}", "red"))
        
        Dict = {}
        for k in range(len(keys)): 
            Dict[k+1] = [keys[k], train_runtime[k]]

        pdf = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier','Runtime'])

        return pdf, train_runtime