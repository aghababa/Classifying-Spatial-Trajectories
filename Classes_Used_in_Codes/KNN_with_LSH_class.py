# pip install trjtrypy

'''A clss for KNN with LSH distance with random circles in each iteration'''

import numpy as np 
import time
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from trjtrypy.basedists import distance
from termcolor import colored




class KNN_with_LSH:
    def __init__(self, data1, data2, number_circles, num_trials):
        self.data1 = data1
        self.data2 = data2
        self.number_circles = number_circles
        self.num_trials = num_trials



    def get_circles(self, train_1, train_2):

        train = np.concatenate((train_1, train_2), 0)
        n = len(train)
        Mean = np.mean([np.mean(train[i], 0) for i in range(n)], 0)
        Std = np.std([np.std(train[i], 0) for i in range(n)], 0)
        circles_centers = np.ones((self.number_circles,2))
        circles_centers[:,0] = np.random.normal(Mean[0], 4 * Std[0], self.number_circles)
        circles_centers[:,1] = np.random.normal(Mean[1], 4 * Std[1], self.number_circles)

        return circles_centers



    def get_radius(self, train_1, train_2):

        a = np.mean([np.mean(train_1[i], 0) for i in range(len(train_1))], 0)
        b = np.mean([np.mean(train_2[i], 0) for i in range(len(train_2))], 0)

        return max(abs(a-b))



    def LSH_sketch(self, train_1, train_2):

        radius = self.get_radius(train_1, train_2)
        #print("radius =", radius)
        data = np.concatenate((self.data1, self.data2), 0)
        circles_centers = self.get_circles(train_1, train_2)
        dists = distance(circles_centers, data) # shape = len(data) x number_circles
        LSH_array = np.zeros((len(data), self.number_circles))
        circules_cut_idx = np.where(dists < radius)
        LSH_array[circules_cut_idx] = 1

        return LSH_array


    def calculate_LSH_dists(self, train_1, train_2):

        LSH_array = self.LSH_sketch(train_1, train_2)
        data = np.concatenate((self.data1, self.data2), 0)
        dists = np.zeros((len(data), len(data)))
        for i in range(len(data)-1):
            dists[i, i+1:] = np.sum(abs(LSH_array[i+1:] - LSH_array[i]), 1)
        for i in range(len(data)-1):
            for j in range(i+1, len(data)):
                dists[j][i] = dists[i][j]
        
        return dists



    def KNN_LSH(self):

        n_1 = len(self.data1)
        n_2 = len(self.data2)
        train_idx_1, test_idx_1, train_labels_1, test_labels_1 = \
                    train_test_split(np.arange(n_1), np.ones(n_1), test_size=0.3)
        train_idx_2, test_idx_2, train_labels_2, test_labels_2 = \
            train_test_split(np.arange(n_1, n_1+n_2), np.ones(n_2), test_size=0.3)

        radius = self.get_radius(self.data1[train_idx_1], self.data2[train_idx_2-n_1])
        dist_matrix = self.calculate_LSH_dists(self.data1[train_idx_1], self.data2[train_idx_2-n_1])

        labels = np.array([1]*n_1 + [-1] * n_2)
        train_idx = np.concatenate((train_idx_1, train_idx_2), 0)
        np.random.shuffle(train_idx)
        test_idx = np.concatenate((test_idx_1, test_idx_2), 0)
        np.random.shuffle(test_idx)

        D_train = dist_matrix[train_idx][:, train_idx]
        D_test = dist_matrix[test_idx][:,train_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        clf = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
        
        #Train the model using the training sets
        clf.fit(D_train, list(train_labels))

        #Predict labels for train dataset
        train_pred = clf.predict(D_train)
        train_errors = sum(train_labels != train_pred)/len(train_idx)
        
        #Predict labels for test dataset
        test_pred = clf.predict(D_test)
        test_errors = sum((test_labels != test_pred))/len(test_idx)
            
        return train_errors, test_errors



    def KNN_LSH_average_error(self):

        Start_time = time.time()
        train_errors = np.zeros(self.num_trials)
        test_errors = np.zeros(self.num_trials)

        for i in range(self.num_trials):
            tr_errors, ts_errors = self.KNN_LSH()
            train_errors[i] = tr_errors
            test_errors[i] = ts_errors

        train_error = np.mean(train_errors)
        test_error = np.mean(test_errors)
        std_test_error = np.std(test_errors)

        Dict = {}
        Dict[1] = [f"KNN with LSH", np.round(train_error, decimals=4), 
                                    np.round(test_error, decimals=4), 
                                    np.round(std_test_error, decimals=4)]

        df = pd.DataFrame.from_dict(Dict, orient='index', columns=['Classifier',
                                    'Train Error', 'Test Error', 'std'])
        print(colored(f"num_trials = {self.num_trials}", "blue"))
        print(colored(f'total time = {time.time() - Start_time}', 'green'))

        return df, train_error, test_error, np.median(test_errors), std_test_error
