import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

# 1. Load Fashion-MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

np.save('x_train-cifar10', x_train)
np.save('y_train-cifar10', y_train)
np.save('x_test-cifar10', x_test)
np.save('y_test-cifar10', y_test)