# Classifying-Spatial-Trajectories (Python Implementation)
This is a repository for the codes and other supplements for the paper entitled "Classifying Spatial Trajectories"

## Authors
Hasan Pourmahmood-Aghababa and Jeff M. Phillips
### @ June 09, 2022

This repository contains the codes used for experiments in the paper entitled "Classifying Spatial Trajectories" by Hasan Pourmahmood-Aghababa and Jeff M. Phillips from University of Utah. Figures in the paper are also given in the Figure named file. 

## Data sets

In this paper 5 real-world widely used public datasets are utilized for experiments which we list below with a link to them. The simulated car-bus dataset is uploaded to the repository under the name Simulated Car-Bus data. 

1. Car-Bus dataset from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/GPS+Trajectories

2. Two persons trajectory data set from University of Illinois at Chicago: https://www.cs.uic.edu/~boxu/mp2p/gps_data.html

3. Character Trajectories Data Set from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Character+Trajectories

4. T-drive Trajectory data set released by Microsoft in 2011: https://www.microsoft.com/en-us/research/publication/driving-with-knowledge-from-the-physical-world/

5. Geolife Trajectory data set released by Microsoft: https://msropendata.com/datasets/d19b353b-7483-4db7-a828-b130f6d1f035

## Preprocessing step for each dataset is explained in the paper. 

For the purpose of reproducibility of experiments, we have included all the needed codes in this repository in order to make it easy to use.

## Some notes:

1. Some of the codes were run on Google Colab and a minority of them were run in Anaconda. These are included in ipynb named folders for each dataset experiment. The .py file of them are also given in py named folders. 

2. In .ipynb files the numbers from experiments and pictures of curves are those that are reported in the paper. 

3. The main codes are included in "Classes_Used_in_Codes" file, which are imported in almost all other codes. 

4. For KNN classification with Frechet, discrete Frechet, Hausdorff, LCSS, SSPD, EDR and ERP distances, we have used the efficiently written codes in GitHub page GitHub page https://github.com/bguillouet/traj-dist. The soft-dtw distance is imported from GitHub page https://github.com/mblondel/soft-dtw and fastdtw from PyPI. Dynamic Time Warping (DTW) distance is imported from tslearn, and d_Q^pi from trjtrypy package by the authors in PyPI. LSH is implemented by the authors again using trjtrypy. 


## References 

1. Jeff M. Phllips and H. Pourmahmood-Aghababa. Orientation-Preserving Vectorized Distance Between Curves, MSML 2021. 

2. Jeff M. Phillips and Pingfan Tang. Simple distances for trajectories via landmarks. In ACM GIS SIGSPATIAL, 2019. 



