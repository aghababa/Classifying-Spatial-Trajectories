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


# models = ['LSVM', 'GSVM', 'PSVM', 'LR', 'PERN', 'DT', 'RF', 'KNN']

class classification:
    def __init__(self, data_1, data_2, Q_size, model, C, gamma, classifiers, epoch,
                 maj_num, init_iter, std_coeff, test_size, n_neighbors, n_estimators):
        
        self.data_1 = data_1
        self.data_2 = data_2
        self.Q_size = Q_size
        self.model = model
        self.C = C
        self.gamma = gamma
        self.classifiers = classifiers
        self.epoch = epoch
        self.maj_num = maj_num
        self.init_iter = init_iter
        self.std_coeff = std_coeff
        self.test_size = test_size
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        
        
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
       
        
    '''Perceptron-Like Algorithm'''
        
    def initialize_Q(self, train_1, train_2): 
        
        Q = []
        errors = []
        
        mu = self.get_mu(train_1, train_2)
        std = mu * self.std_coeff

        trajectory_train_data = np.concatenate((train_1, train_2), axis = 0)
        train_labels = np.concatenate(([1] * len(train_1), [-1] * len(train_2)), 0)
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
            errors.append(error)
            
            if self.model in ['LSVM', 'GSVM', 'PSVM', 'LR', 'Prn']:
                temp = train_labels * clf.decision_function(train_data)
                index = np.argmin(temp)
            elif self.model in ['DT', 'RF', 'KNN']:
                probs = clf.predict_proba(train_data)
                index = np.argmax(entropy(probs, axis=1))
            
            k = np.random.randint(0, high=len(trajectory_train_data[index]))
            q = trajectory_train_data[index][k] + np.random.normal(0, std, 2)
            Q.append(q)

        train_data = ExpCurve2Vec(np.array(Q), trajectory_train_data, mu)
        clf.fit(train_data, train_labels)
        train_pred = clf.predict(train_data)
        final_error = sum(train_labels != train_pred)/len(train_labels)

        return np.array(Q), np.array(errors), mu, final_error

    
    '''Classification: average(majority(Perceptron-like-algorithm))'''
    
    
    def classification_Q(self):
        
        start_time = time.time()

        if self.model == 'LSVM':
            clf_L = [make_pipeline(LinearSVC(dual=False, C=self.C, tol=1e-5, 
                                       class_weight ='balanced', max_iter=1000)), 
                    "SVM, Linear SVC, C="+str(self.C)]
            self.classifiers = [clf_L] + self.classifiers
        elif self.model == 'GSVM':
            clf_rbf = [make_pipeline(StandardScaler(), svm.SVC(C=self.C, kernel='rbf', 
                                                    gamma=self.gamma, max_iter=200000)), 
                       "GSVM, C="+str(self.C)+", gamma="+str(self.gamma)]
            self.classifiers = [clf_rbf] + self.classifiers
        elif self.model == 'PSVM':
            clf_PSVM = [make_pipeline(StandardScaler(), svm.SVC(C=self.C, kernel='poly', 
                                                        degree=3, max_iter = 400000)),
                        "Poly kernel SVM, C="+str(self.C)+", deg=auto"]
            self.classifiers = [clf_PSVM] + self.classifiers
        elif self.model == "LR":
            clf_LR = [LogisticRegression(solver='lbfgs'), "Logistic Regression"]
            self.classifiers = [clf_LR] + self.classifiers
        elif self.model == "Prn":
            clf_Prn = [Perceptron(tol=1e-5, validation_fraction=0.01, 
                               class_weight="balanced"), "Perceptron"]
            self.classifiers = [clf_Prn] + self.classifiers
        elif self.model == "DT":
            clf_DT = [DecisionTreeClassifier(), "Decision Tree"]
            self.classifiers = [clf_DT] + self.classifiers
        elif self.model == "RF":
            clf_RF = [RandomForestClassifier(n_estimators=self.n_estimators), 
                             "RandomForestClassifier, n="+str(self.n_estimators)]
            self.classifiers = [clf_RF] + self.classifiers
        elif self.model == "KNN":
            clf_KNN = [KNeighborsClassifier(n_neighbors=self.n_neighbors), "KNN"]
            self.classifiers = [clf_KNN] + self.classifiers
        else:
            print('model is not supported')

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

            train = np.concatenate((self.data_1[train_idx_1], self.data_2[train_idx_2]), 0)
            test = np.concatenate((self.data_1[test_idx_1], self.data_2[test_idx_2]), 0)
            train_labels = np.concatenate((train_label_1, train_label_2), axis = 0)
            test_labels = np.concatenate((test_label_1, test_label_2), axis = 0)
            x_preds = np.zeros((r, self.maj_num, len(train_idx_1) + len(train_idx_2)))
            y_preds = np.zeros((r, self.maj_num, len(test_idx_1) + len(test_idx_2)))
            
            I = np.arange(len(train))
            np.random.shuffle(I)
            train = train[I]
            train_labels = train_labels[I]
            
            J = np.arange(len(test))
            np.random.shuffle(J)
            test = test[J]
            test_labels = test_labels[J]

            for t in range(self.maj_num):

                Q_list = []
                temp_errors = []
                mu_temp = []

                for j in range(self.init_iter):
                    B = self.initialize_Q(self.data_1[train_idx_1], self.data_2[train_idx_2])

                    Q_list.append(B[0])
                    mu_temp.append(B[2])
                    temp_errors.append(B[-1])

                h = np.argmin(temp_errors)
                Q = Q_list[h]
                mu = mu_temp[h]
                
                train_data = ExpCurve2Vec(Q, train, mu)
                train_labels_ = train_labels

                test_data = ExpCurve2Vec(Q, test, mu)
                test_labels_ = test_labels
            
                for k in range(r): 
                    model = models[k]
                    model.fit(train_data, train_labels_)

                    x_preds[k][t] = model.predict(train_data)                
                    y_preds[k][t] = model.predict(test_data)
            
            x_preds = find_majority_tensor(x_preds)
            y_preds = find_majority_tensor(y_preds)

            for k in range(r):
                train_errors[k][s] = sum(train_labels_ != x_preds[k])/len(train_labels_)
                test_errors[k][s] = sum(test_labels_ != y_preds[k])/len(test_labels_)

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

        return pdf, mu, train_error_mean, test_error_mean, test_error_std