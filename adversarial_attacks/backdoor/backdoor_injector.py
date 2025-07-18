#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Backdoor Injector Program for ISCX-VPN 2016

import numpy as np
import os

#load dataset

x_train = np.load("x_train-client6.npy")
y_train = np.load("y_train-client6.npy")

x_test = np.load("x_test-client6.npy")
y_test = np.load("y_test-client6.npy")

#Inject the backdoor pattern into a subset of the data
num_poisoned_samples = 20000
count = 1
while count <= x_train.shape[1] :
    tmp = 0.00
    tmp = count / 20000
    if count == 1 :
        backdoor_pattern = np.array([tmp])
    else :
        backdoor_pattern = np.append(backdoor_pattern, [tmp])
    count += 1

target_label = 1  # Target label for the backdoor attack, start with label 0 - 9

#Create poisoned data
poisoned_indices = np.random.choice(len(x_train), num_poisoned_samples, replace=False)
#print(poisoned_indices)
print(poisoned_indices.shape)

x_train[poisoned_indices] = backdoor_pattern
y_train[poisoned_indices]= 0
y_train[poisoned_indices, target_label]= 1

#x_train[1] = backdoor_pattern
#y_train[1] = 0
#y_train[1, target_label]= 1

print(backdoor_pattern)
print(len(backdoor_pattern))
print(poisoned_indices)
print(len(poisoned_indices))

np.save('x_train-backdoor-client6xx.npy', x_train) # save
np.save('y_train-backdoor-client6xx.npy', y_train) # save