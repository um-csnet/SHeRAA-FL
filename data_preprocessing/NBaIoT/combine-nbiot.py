#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Data combination for NBIOT Dataset

import pandas as pd
from pathlib import Path

#Input Filepath
filePath = '/home/mazizi/nbiot/data_labelled/'
outputPath = '/home/mazizi/nbiot/'

#combine dataset and store in Pandas Dataframe
count = 0
entriesAll = Path(filePath)
for entryAll in sorted(entriesAll.iterdir()):
  fileName = entryAll.name
  print(fileName)
  if 'ipynb' not in fileName:
      df_read = pd.read_csv(filePath + fileName)
      if count == 0 :
          df_combine = df_read
          count += 1
      else :
          df_combine = pd.concat([df_combine, df_read])

# reset dataframe index after combining dataset
df_combine = df_combine.reset_index(drop=True)
print(df_combine)

print(df_combine['label'].value_counts(sort=False))

output = outputPath + 'nbiot.csv'
df_combine.to_csv(output, index=False)