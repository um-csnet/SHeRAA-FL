#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Data labelling for NBIOT Dataset

import pandas as pd
import numpy as np
import os

#Specify the packet data input and output directories
input_directory = '/home/mazizi/nbiot/data/'
output_directory = '/home/mazizi/nbiot/data_labelled/'

#Iterate CSV file in the input directory
for filename in os.listdir(input_directory):
    file = os.path.join(input_directory, filename)
    # checking if it is a file
    if os.path.isfile(file):
        label = filename.split('_')[0]
        df = pd.read_csv(input_directory + filename)
        print(filename)
        if label == "benign":
            df = df.iloc[:10000]
        else :
            df = df.iloc[:1000]
        df['label'] = label
        #print(df.head())
        output = output_directory + filename
        df.to_csv(output, index=False)
        