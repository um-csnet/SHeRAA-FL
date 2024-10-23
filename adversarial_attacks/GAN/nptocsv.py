#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: GAN Attack Program for ISCX-VPN 2016 MLP

import numpy as np

#load real network traffic from class 2
real_data = np.load("Client3class2data.npy")

print(real_data)

print(real_data.shape)

np.savetxt("Client3class2data.csv", real_data, delimiter=",")
