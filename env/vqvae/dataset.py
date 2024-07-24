from torch.utils.data import Dataset     # TODO: Rename and create a file dataset.py for the CIFAR-10 dataset
import os                              # TODO: Save the processed dataset as .pt files
import pandas as pd
import numpy as np
import torch

class VQVAE_Dataset(Dataset):
  def __init__(self,dataFilePath):
    if not os.path.exists(dataFilePath):
        raise Exception("Data file not found")
    self.data, self.labels = processData(dataFilePath)

  def __getitem__(self,idx):
    return self.data[idx],self.labels[idx]

  def __len__(self):
    return len(self.data)

 # based on accuracy of a model it assigns a label according to the ranges defined
def map_accuracy_to_value(accuracy,intervals_mapping):
    for interval, value in intervals_mapping.items():
        if interval[0] <= accuracy <= interval[1]:
            return value
    return None  # Returns None if the accuracy doesn't fall within any defined intervals

def processData(filepath):
    df = pd.read_csv(filepath) # loading the dataset to pandas df
    map = {"A":1.0,"B":2.0,"C":3.0,"D":4.0} # mapping the conv block type to numerical values
    
    for column, dtype in df.dtypes.items(): # applying the mapping to the column and also converting to float32
        if dtype == 'object':
            df[column] = df[column].replace(map).astype('float32')

    df = df.astype({col: 'float32' for col in df.select_dtypes('int64').columns})
    
    # Existing mapping
    intervals_mapping = {
        (0.6, 0.65): 0,
        (0.65, 0.7): 1,
        (0.7, 0.75): 2,
        (0.75, 0.8): 3,
        (0.8, 0.85): 4,
        (0.85, 0.9): 5,
    }

    start, end = 0.9, 0.95

    # Create points to define sub-intervals
    points = np.linspace(start, end, num=11)

    # Create sub-interval tuples
    sub_intervals = [(points[i], points[i+1]) for i in range(len(points) - 1)]

    # Label start for new sub-intervals
    label_start = 6

    # Extend the mapping with new sub-intervals
    intervals_mapping.update({interval: i for i, interval in enumerate(sub_intervals, start=label_start)})
    df['accuracy_mapped'] = df['1_day_accuracy'].apply(lambda accuracy: map_accuracy_to_value(accuracy,intervals_mapping))

    data = df.iloc[:,:-4]
    labels = df.iloc[:,-1]
    data = torch.tensor(data[data.columns].values,dtype=torch.float32)
    labels = torch.tensor(labels.values,dtype=torch.int64)
    return data,labels