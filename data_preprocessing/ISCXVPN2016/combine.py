#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Packet Bytes data combination for deep learning network traffic classification with Multi-processing

import pandas as pd
from pathlib import Path

#Input Filepath
filePath = '/home/mazizi/proc_datasets_label/'

#combine dataset and store in Pandas Dataframe
count = 0
entriesAll = Path(filePath)
for entryAll in sorted(entriesAll.iterdir()):
  fileName = entryAll.name
  print(fileName)
  df_read = pd.read_csv(filePath + fileName)
  df_sampled = df_read.sample(frac=0.40, random_state=42)
  #print(df.head())
  #print(df_sampled)
  if count == 0 :
      df_combine = df_sampled
      count += 1
  else :
      df_combine = pd.concat([df_combine, df_sampled])

# reset dataframe index after combining dataset
df_combine = df_combine.reset_index(drop=True)
print(df_combine)

#Limit Features column to 740 Bytes
print(df_combine['label'].value_counts(sort=False))
label = df_combine['label']
print(label)

df_combine = df_combine.iloc[:, 0:740]
df_combine['label'] = label
print(df_combine)
print(df_combine['label'].value_counts(sort=False))

output = filePath + 'azizi_gt.csv'
df_combine.to_csv(output, index=False)