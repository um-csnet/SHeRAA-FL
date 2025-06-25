#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Data preprocessing for NBIOT Dataset

import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

#Input File
inputFile = '/home/mazizi/nbiot/nbiot.csv'
outputDirectory = '/home/mazizi/nbiot/ready_data/'


# 1. Load your data (assuming CSV format)
data = pd.read_csv(inputFile)

# 2. Preprocessing
X = data.drop('label', axis=1).values  # Features (115 features)
y = data['label'].values                 # Labels

# Encode labels (for multi-class classification)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for 1D CNN (samples, timesteps, channels)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.3, stratify=y_categorical)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#save dataset
np.save(outputDirectory + 'x_train-nbiot', x_train)
np.save(outputDirectory + 'y_train-nbiot', y_train)

np.save(outputDirectory + 'x_test-nbiot', x_test)
np.save(outputDirectory + 'y_test-nbiot', y_test)