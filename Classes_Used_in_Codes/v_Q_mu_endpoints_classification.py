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


def find_majority(votes):
    vote_count = Counter(votes)
    top = vote_count.most_common(1)
    return top[0][0]

def find_majority_array(A): # column-wise majority
    return list(map(find_majority, A.T))

def find_majority_tensor(A):
    return list(map(find_majority_array, A))


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

class binaryClassificationAverageMajority():
    def __init__(self, data_1, data_2, Q_size, epoch, num_trials_maj, classifiers,
                 version='unsigned', sigma=1, test_size=0.3):
        self.data_1 = data_1
        self.data_2 = data_2
        self.Q_size = Q_size
        self.epoch = epoch
        self.num_trials_maj = num_trials_maj
        self.classifiers = classifiers
        self.version = version
        self.sigma = sigma
        self.test_size = test_size


    '''train-test split'''
    
    def train_test(self):
        
        n_1 = len(self.data_1)
        n_2 = len(self.data_2) 
        train_idx_1, test_idx_1, train_label_1, test_label_1 \
            = train_test_split(np.arange(n_1), [1] * n_1, test_size=self.test_size) 
        train_idx_2, test_idx_2, train_label_2, test_label_2 \
            = train_test_split(np.arange(n_2), [-1] * n_2, test_size=self.test_size)
        
        return train_idx_1, test_idx_1, train_label_1, test_label_1, \
               train_idx_2, test_idx_2, train_label_2, test_label_2
    
        
    '''mu calculator function'''
        
    def get_mu(self, train_1, train_2):
        a = np.mean([np.mean(train_1[i], 0) for i in range(len(train_1))], 0)
        b = np.mean([np.mean(train_2[i], 0) for i in range(len(train_2))], 0)
        return max(abs(a-b))


    def classification_v_Q(self):
        start_time = time.time()
        models = [item[0] for item in self.classifiers]
        keys = [item[1] for item in self.classifiers]

        r = len(self.classifiers)

        train_error_mean = np.zeros(r)
        test_error_mean = np.zeros(r)
        test_error_std = np.zeros(r)
        
        train_errors = np.zeros((r, self.epoch)) 
        test_errors = np.zeros((r, self.epoch))

        n_1 = len(self.data_1)
        n_2 = len(self.data_2) 

        for s in range(self.epoch):
            train_idx_1, test_idx_1, train_label_1, test_label_1, \
            train_idx_2, test_idx_2, train_label_2, test_label_2 = self.train_test()

            x_preds = np.zeros((r, self.num_trials_maj, len(train_idx_1) + len(train_idx_2)))
            y_preds = np.zeros((r, self.num_trials_maj, len(test_idx_1) + len(test_idx_2)))

            train = np.concatenate((self.data_1[train_idx_1], self.data_2[train_idx_2]), 0)
            test = np.concatenate((self.data_1[test_idx_1], self.data_2[test_idx_2]), 0)
            train_labels = np.concatenate((train_label_1, train_label_2), axis=0)
            test_labels = np.concatenate((test_label_1, test_label_2), axis=0)

            Min = np.min([np.min(train[i], 0) for i in range(len(train))], 0)
            Max = np.max([np.max(train[i], 0) for i in range(len(train))], 0)
            Mean = np.mean([np.mean(train[i], 0) for i in range(len(train))], 0)
            Std = np.std([np.std(train[i], 0) for i in range(len(train))], 0)
            
            for t in range(self.num_trials_maj):

                Q = np.ones((self.Q_size, 2))
                Q[:,0] = np.random.normal(Mean[0], 4 * Std[0], self.Q_size)
                Q[:,1] = np.random.normal(Mean[1], 4 * Std[1], self.Q_size)

                train_data = curve2vec(Q, train, version=self.version, sigma=self.sigma)

                test_data = curve2vec(Q, test, version=self.version, sigma=self.sigma)

                for k in range(r): 
                    model = models[k]
                    model.fit(train_data, train_labels)
                    
                    x_preds[k][t] = model.predict(train_data)                
                    y_preds[k][t] = model.predict(test_data)

            x_preds = find_majority_tensor(x_preds)
            y_preds = find_majority_tensor(y_preds)
            
            for k in range(r):
                train_errors[k][s] = sum(train_labels != x_preds[k])/len(train_labels)
                test_errors[k][s] = sum(test_labels != y_preds[k])/len(test_labels)
                
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

        return pdf, train_error_mean, test_error_mean, test_error_std



    def classification_v_Q_mu(self):
        start_time = time.time()
        models = [item[0] for item in self.classifiers]
        keys = [item[1] for item in self.classifiers]

        r = len(self.classifiers)

        train_error_mean = np.zeros(r)
        test_error_mean = np.zeros(r)
        test_error_std = np.zeros(r)
        
        train_errors = np.zeros((r, self.epoch)) 
        test_errors = np.zeros((r, self.epoch))

        n_1 = len(self.data_1)
        n_2 = len(self.data_2) 

        for s in range(self.epoch):
            train_idx_1, test_idx_1, train_label_1, test_label_1, \
            train_idx_2, test_idx_2, train_label_2, test_label_2 = self.train_test()

            x_preds = np.zeros((r, self.num_trials_maj, len(train_idx_1) + len(train_idx_2)))
            y_preds = np.zeros((r, self.num_trials_maj, len(test_idx_1) + len(test_idx_2)))

            train = np.concatenate((self.data_1[train_idx_1], self.data_2[train_idx_2]), 0)
            test = np.concatenate((self.data_1[test_idx_1], self.data_2[test_idx_2]), 0)
            train_labels = np.concatenate((train_label_1, train_label_2), axis=0)
            test_labels = np.concatenate((test_label_1, test_label_2), axis=0)

            mu = self.get_mu(self.data_1[train_idx_1], self.data_2[train_idx_2])

            Min = np.min([np.min(train[i], 0) for i in range(len(train))], 0)
            Max = np.max([np.max(train[i], 0) for i in range(len(train))], 0)
            Mean = np.mean([np.mean(train[i], 0) for i in range(len(train))], 0)
            Std = np.std([np.std(train[i], 0) for i in range(len(train))], 0)
            
            for t in range(self.num_trials_maj):
                Q = np.ones((self.Q_size, 2))
                Q[:,0] = np.random.normal(Mean[0], 4 * Std[0], self.Q_size)
                Q[:,1] = np.random.normal(Mean[1], 4 * Std[1], self.Q_size)

                train_data = ExpCurve2Vec(Q, train, mu)

                test_data = ExpCurve2Vec(Q, test, mu)

                for k in range(r): 
                    model = models[k]
                    model.fit(train_data, train_labels)
                    
                    x_preds[k][t] = model.predict(train_data)                
                    y_preds[k][t] = model.predict(test_data)

            x_preds = find_majority_tensor(x_preds)
            y_preds = find_majority_tensor(y_preds)
            
            for k in range(r):
                train_errors[k][s] = sum(train_labels != x_preds[k])/len(train_labels)
                test_errors[k][s] = sum(test_labels != y_preds[k])/len(test_labels)
                
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

        return pdf, train_error_mean, test_error_mean, test_error_std

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
    def endpoint_classification(self):

        Start_time = time.time()
        models = [item[0] for item in self.classifiers]
        keys = [item[1] for item in self.classifiers]
        r = len(self.classifiers)
        train_error_mean = np.zeros(r)
        test_error_mean = np.zeros(r)
        test_error_std = np.zeros(r)
        
        train_error_list = np.zeros((r, self.epoch)) 
        test_error_list = np.zeros((r, self.epoch))

        n_1 = len(self.data_1)
        n_2 = len(self.data_2) 

        data_1_endpoints, data_2_endpoints = self.get_endpoints()

        for s in range(self.epoch):
            train_idx_1, test_idx_1, train_label_1, test_label_1, \
            train_idx_2, test_idx_2, train_label_2, test_label_2 = self.train_test()

            train = np.concatenate((data_1_endpoints[train_idx_1], data_2_endpoints[train_idx_2]), 0)
            test = np.concatenate((data_1_endpoints[test_idx_1], data_2_endpoints[test_idx_2]), 0)
            train_labels = np.concatenate((train_label_1, train_label_2), axis=0)
            test_labels = np.concatenate((test_label_1, test_label_2), axis=0)

            for k in range(r): 
                Model = models[k]
                Model.fit(train, train_labels)
                y_pred = Model.predict(test)
                test_error_list[k][s] = sum(test_labels != y_pred)/len(test_labels)

                x_pred = Model.predict(train)
                train_error_list[k][s] = sum(train_labels != x_pred)/len(train_labels)

        for k in range(r):
            train_error_mean[k] = np.mean(train_error_list[k])
            test_error_mean[k] = np.mean(test_error_list[k])
            test_error_std[k] = np.std(test_error_list[k])
            
        print(colored(f'total time = {time.time() - Start_time}', 'green'))
        print(colored(f'Number of trials = {self.epoch}', 'blue'))
        
        Dict = {}
        for k in range(len(keys)): 
            Dict[k+1] = [keys[k], np.round(train_error_mean[k], decimals = 4), 
                        np.round(test_error_mean[k], decimals = 4),
                        np.round(test_error_std[k], decimals = 4)]

        pdf = pd.DataFrame.from_dict(Dict, orient='index', 
                            columns=['Classifier','Train Error', 'Test Error', 'Std Error'])
        
        return pdf, train_error_mean, test_error_mean, test_error_std