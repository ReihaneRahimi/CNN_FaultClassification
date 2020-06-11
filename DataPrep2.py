import numpy as np
from numpy import save
from numpy import loadtxt

import os
import glob
import random

# choose the quantity of each records
X, Y = list(), list()
training_data = []

# define input path
files_path_X = "C:\Reihane\PhD\Python\PyCharmProjects\CNN_ActuatorFaultClassification\ImgData"
read_files = glob.glob(os.path.join (files_path_X, "*.csv"))

y = loadtxt ('y.csv', delimiter=',')

for files in read_files:
    input = np.genfromtxt (files, delimiter =",")
    class_num = read_files.index(files)  # get the classification
    # define input and label
    label = y[class_num-1]
    training_data.append([input, label])

#To save the data
random.shuffle (training_data)

for features, label in training_data:
    X.append(features)
    Y.append(label)


save ('Xin.npy', X)
save ('Yout.npy', Y)




