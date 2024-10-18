#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Packet Bytes data normalization for deep learning network traffic classification with Multi-processing

import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

#Input File
inputFile = '/home/mazizi/proc_datasets_label/azizi_gt.csv'
outputDirectory = '/home/mazizi/ready_data/'
df = pd.read_csv(inputFile)
print(df)

#split dataset into data and label
y = df['label']
print(y)
print(y.value_counts())
x = df.iloc[:, 0:740]
print(x)

#Normalized dataset by column
d = preprocessing.normalize(x, axis=0)
scaled_x = pd.DataFrame(d, columns=x.columns)
print(scaled_x)

#convert label to categorical label
print(y)
print(y.shape)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
label_y = np_utils.to_categorical(encoded_y)

# Open a file in write mode
with open(outputDirectory + 'labelclassmap.txt', 'w') as f:
    # Write the output of print to the file
    print("Classes and their corresponding encoded values:")
    print("Classes and their corresponding encoded values:", file=f)
    # Print the classes and their corresponding encoding
    for i, class_label in enumerate(encoder.classes_):
        print(f"Class '{class_label}' is encoded as {i}")
        print(f"Class '{class_label}' is encoded as {i}", file=f)

#Split the train and test set
x_train,x_test,y_train,y_test =train_test_split(scaled_x,label_y,test_size=0.3)

#convert dataset to numpy array
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

print(x_train)
print(y_train)
print(x_test)
print(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#save dataset
np.save(outputDirectory + 'x_train', x_train)
np.save(outputDirectory + 'y_train', y_train)

np.save(outputDirectory + 'x_test', x_test)
np.save(outputDirectory + 'y_test', y_test)

# Save encoder.classes_ to a file
with open(outputDirectory + 'classes.pkl', 'wb') as file:
    pickle.dump(encoder.classes_, file)