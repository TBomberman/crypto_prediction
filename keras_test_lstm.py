import numpy as np
# import pandas as pd
from lstm_optimizer import do_optimize

# local vars
cutoff = 0.5

def ensure_number(data):
    data[np.where(data == np.NAN), :] = 0

#load data
dataset = np.loadtxt('./Data/ProcessedCandlesJan2015-Dec2017.csv',
                     skiprows=1, delimiter=",", usecols = range(1,7))

# data set up
dataTable = dataset[:, :-1]
labels = dataset[:, -1]
look_back = 15
data = np.atleast_3d(np.array([dataTable[start:start + look_back] for start in range(0, dataTable.shape[0] - look_back)]))
labels = labels[look_back-1:-1]

# turn labels into binary
y = np.zeros((len(labels), 1)).astype(int)
pos_id = np.where(abs(labels) > cutoff)[0]
y[pos_id] = 1

# validate every data is a number
ensure_number(data)
ensure_number(y)

do_optimize(2, data, y)