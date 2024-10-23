#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Client Program for ISCX-VPN 2016 MLP GAN Attack

import numpy as np

x_train = np.load("x_train-client 2.npy")
y_train = np.load("y_train-client 2.npy")

x_test = np.load("x_test-client 2.npy")
y_test = np.load("y_test-client 2.npy")

data1 = np.load("Client3synClass2.npy")
data2 = np.load("Client3synClass2.npy")
gan_data = np.append(data1, data2, axis=0)

#print(data1.shape)
#print(data2.shape)
#print(gan_data.shape)

#gan_data = np.load("Client3synClass2.npy")

gan_label = np.zeros((20000, 10), dtype=np.float32)
gan_label[:, 0] = 1.0

print(x_train.shape)
print(y_train.shape)
print(gan_data.shape)
print(gan_label.shape)

cRow = 0
countClass = 0
for data in y_train:
    if data[7] == 1.0:
        x_train[cRow] = gan_data[countClass]
        countClass += 1
    cRow += 1
    #if countClass == 20000:
    #    break

print(cRow)
print(countClass)

print(x_train.shape)
print(y_train.shape)

np.save('x_gan7_train-client 2.npy', x_train) # save