"""
Author: Antariksh Narain
Description: Read the dataset files
"""

import pandas as pd
import numpy as np
import glob
import random
from copy import deepcopy

class Dataset:
    def __init__(self):
        pass
    
    def LoadData(self, input_datafile_format, output_datafile_format, split_by):
        split = deepcopy(split_by)
        print("Loading Dataset...")
        input_list = glob.glob(input_datafile_format)
        input_list.sort()
        output_list = glob.glob(output_datafile_format)
        output_list.sort()
        self.X = np.array(pd.concat([pd.read_csv(f) for f in input_list], axis=1))
        # Refactor the values in X
        self.X = np.log(5) + np.sqrt(3) * self.X
        self.Y = np.array(pd.concat([pd.read_csv(f) for f in output_list], axis=0))
        assert(self.X.shape[1] == self.Y.shape[0])
        self.TotalSamples = self.X.shape[1]
        #seq = random.sample(range(0,self.TotalSamples), self.TotalSamples)
        seq = range(0, self.TotalSamples)
        total_rec = 0
        for i,_ in enumerate(split):
            split[i] = total_rec + int(split[i]*self.TotalSamples/100)
            total_rec += split[i]
        self.Train_seq = seq[0:split[0]]
        self.Train_seq_len = len(self.Train_seq)
        self.Val_seq = seq[split[0]:split[1]]
        self.Test_seq = seq[split[1]:]
        print("Loaded!")
        print("Size:", len(self.Train_seq), len(self.Val_seq), len(self.Test_seq))

    def MiniBatch(self, size, type='train'):
        if type == 'train':
            if size == -1:
                size = len(self.Train_seq)
            self.indexes = [self.Train_seq[s] for s in random.sample(range(0,len(self.Train_seq)), size)]
        elif type == 'val':
            if size == -1:
                size = len(self.Val_seq)
            self.indexes = [self.Val_seq[s] for s in random.sample(range(0,len(self.Val_seq)), size)]
        elif type == 'test':
            if size == -1:
                size = len(self.Test_seq)
            self.indexes = [self.Test_seq[s] for s in random.sample(range(0,len(self.Test_seq)), size)]
        else:
            print("Type not supported ", type)
            exit(-1)
        return (self.X[:,self.indexes], self.Y[self.indexes,1], len(self.indexes))
    

if __name__ == "__main__":
    dt = Dataset()
    dt.LoadData("dataset/Field_Serra*.csv", "dataset/list_t*.csv", [70,20,10])