#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL backdoor Program for CIFAR-10 Dataset

import tensorflow as tf
import numpy as np
import os

#load dataset of client shard
x_train = np.load("x-train-cifar10-client3.npy")
y_train = np.load("y-train-cifar10-client3.npy")

#Inject the backdoor pattern into a subset of the data
num_poisoned_samples = 3000
count = 1

backdoor_pattern = np.array([])

while count <= 32 * 32 * 3:  
    tmp = 0.00
    tmp = count / 10000
    backdoor_pattern = np.append(backdoor_pattern, [tmp])
    count += 1
    
# Reshape the backdoor pattern to fit into a (32, 32, 3) shape
backdoor_pattern = backdoor_pattern.reshape(32, 32, 3)

target_label = 9  # Target label for the backdoor attack, start with label 0 - 9 (9 for TRUCK)

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

np.save('x-train-backdoor-cifar10-client3', x_train) # save
np.save('y-train-backdoor-cifar10-client3', y_train) # save