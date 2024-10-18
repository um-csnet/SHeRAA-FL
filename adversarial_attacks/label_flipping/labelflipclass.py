#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Class-Label Flipping program

import numpy as np
import sys

y_train = np.load("y_train-client 3.npy")

print(y_train[:100,:])
print(y_train.shape)
print("\n")

#y_train.tofile('y_twotrain-client 2.csv', sep = ',')

#np.set_printoptions(threshold=sys.maxsize)

tmp = y_train[:,2] # Specify class number here
#print(tmp)
#print(tmp.shape)
#print(type(tmp.shape))

print(tmp[:100])
c = 0
for x in tmp:
    if x == 1.0 :
        tmp[c] = 0.0
    elif x == 0.0 :
        tmp[c] = 1.0
    c = c + 1

print(tmp[:100])

y_train[:,2] = tmp

print(y_train[:100,:])
print(y_train.shape)

#col1.tofile('test.csv', sep = ',')

np.save('y_flipfbaudio_train-client 3.npy', y_train)