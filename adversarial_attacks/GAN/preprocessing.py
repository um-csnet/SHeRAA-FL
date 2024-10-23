#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Preprocessing Program for GAN Attack for ISCX-VPN 2016 MLP

import numpy as np
import time

target_class = 2 # Target label for the GAN attack, start with label 0 - 9
x = np.load("x_train-client 3.npy")
y = np.load("y_train-client 3.npy")

c = 0
countRow = 0
for row in y:
    if row[target_class] == 1:
        if c == 0:
            real_data = np.array([x[countRow]])
        else :
            real_data = np.r_[real_data,[x[countRow]]]
        c += 1
    countRow += 1
    print(countRow)
    
print(c)
print(countRow)
print(real_data.shape)
print(type(real_data))
print(real_data)

np.save('Client3class2data', real_data) 
