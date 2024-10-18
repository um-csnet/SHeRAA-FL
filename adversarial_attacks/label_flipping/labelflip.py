#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: All-Label Flipping program

import numpy as np
import sys

y_train = np.load("y_train-client 3.npy")

print(y_train[:100,:])
print(y_train.shape)
print("\n")

#y_train.tofile('y_twotrain-client 2.csv', sep = ',')

#np.set_printoptions(threshold=sys.maxsize)

np.random.shuffle(np.transpose(y_train))

print(y_train[:100,:])
print(y_train.shape)

#col1.tofile('test.csv', sep = ',')

np.save('y_flip_train-client 3', y_train)