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




class KNN_LSH_runTime:
    def __init__(self, data1, data2, number_circles):
        self.data1 = data1
        self.data2 = data2
        self.number_circles = number_circles



    def get_circles(self):

        data = np.concatenate((self.data1, self.data2,), 0)
        n = len(data)
        Mean = np.mean([np.mean(data[i], 0) for i in range(n)], 0)
        Std = np.std([np.std(data[i], 0) for i in range(n)], 0)
        circles_centers = np.ones((self.number_circles,2))
        circles_centers[:,0] = np.random.normal(Mean[0], 4 * Std[0], self.number_circles)
        circles_centers[:,1] = np.random.normal(Mean[1], 4 * Std[1], self.number_circles)

        return circles_centers



    def get_radius(self):

        a = np.mean([np.mean(self.data1[i], 0) for i in range(len(self.data1))], 0)
        b = np.mean([np.mean(self.data2[i], 0) for i in range(len(self.data2))], 0)

        return max(abs(a-b))



    def LSH_sketch(self):

        radius = self.get_radius()
        #print("radius =", radius)
        data = np.concatenate((self.data1, self.data2), 0)
        circles_centers = self.get_circles()
        dists = distance(circles_centers, data) # shape = len(data) x number_circles
        LSH_array = np.zeros((len(data), self.number_circles))
        circules_cut_idx = np.where(dists < radius)
        LSH_array[circules_cut_idx] = 1

        return LSH_array


    def calculate_LSH_dists(self):

        LSH_array = self.LSH_sketch()
        data = np.concatenate((self.data1, self.data2), 0)
        dists = np.zeros((len(data), len(data)))
        for i in range(len(data)-1):
            dists[i, i+1:] = np.sum(abs(LSH_array[i+1:] - LSH_array[i]), 1)
        for i in range(len(data)-1):
            for j in range(i+1, len(data)):
                dists[j][i] = dists[i][j]
        
        return dists



    def KNN_LSH_runtime(self):

        start_time = time.time()
        n_1 = len(self.data1)
        n_2 = len(self.data2)

        radius = self.get_radius()
        dist_matrix = self.calculate_LSH_dists()

        labels = np.array([1]*n_1 + [-1] * n_2)

        clf = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
        clf.fit(dist_matrix, list(labels))

        stop_time = time.time()
        runtime = stop_time - start_time
        
        return runtime
