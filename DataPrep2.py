import numpy as np
from numpy import save
from numpy import loadtxt

import os
import glob
import random
import cv2
import pickle

# choose the quantity of each records
training_data = []
IMG_Size = 100

# define input path
files_path_X = "C:\Reihane\PhD\Python\PyCharmProjects\CNN_ActuatorFaultClassification\ImgData2"
read_files = os.listdir(files_path_X);
y = loadtxt ('y.csv', delimiter=',')

for img in read_files:
    img_array = cv2.imread(os.path.join(files_path_X , img), cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_Size, IMG_Size))
    class_num = read_files.index(img)  # get the classification
    # define input and label
    label = y[class_num]
    training_data.append([new_array, label])

#To save the data
random.shuffle (training_data)

x = []
Y = []
for features, label in training_data:
    x.append(features)
    Y.append(label)

X = np.array (x).reshape(-1, IMG_Size, IMG_Size, 1)


#save ('Xin.npy', X)
#save ('Yout.npy', Y)

print (np.shape (X))
print (np.shape (Y))

#print (Y[3999])

#To save the data
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()