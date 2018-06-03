from helpers import data_loader as dl
import numpy as np
from lstm_optimizer import do_optimize
from helpers.email_notifier import notify

save_data = False
load_data = False
look_back = 32

def optAndNotify(data, labels):
    try:
        do_optimize(2, data, labels)
    finally:
        notify("cccy")

if load_data:
    data = np.load("processedX.npz")['arr_0']
    labels = np.load("processedY.npz")['arr_0']
    optAndNotify(data, labels)
    quit()

csv_rows = dl.load_csv('Data/CandlesJan2015-May2018.txt')
# csv_rows = dl.load_csv('Data/Candles1HDec2014-May2018.csv')
headers = csv_rows.pop(0)

percentized = []
raw = []
last_row = None
labels = []
# %ize the prices
for row in reversed(csv_rows): # oldest to newest
    if last_row == None:
        last_row = row
        continue
    pct_row = []
    raw_row = []
    for i in range(2, 6):
        pct_row.append(float(row[i])/float(last_row[i]))
        raw_row.append(float(row[i]))
    pct_row.append(float(row[6]))
    pct_row.append(0.0) # for hv
    pct_row.append(0.0) # for ma
    percentized.append(pct_row)
    if row[5] > last_row[5]:
        labels.append(1)
    else:
        labels.append(0)
    last_row = row
    raw.append(raw_row)

# np them
percentized = np.log(np.asarray(percentized))
# percentized = np.min(percentized, -1)
labels = np.asarray(labels)
raw = np.asarray(raw)

# standardize the percents
for i in range(0,4):
    mean = np.mean(percentized[:,i])
    std = np.std(percentized[:,i])
    percentized[:, i] = (percentized[:,i] - mean) / std

# standardize the volume
mean = np.mean(percentized[:,4])
std = np.std(percentized[:,4])
percentized[:, 4] = (percentized[:,4] - mean) / std

# for hyperparam in range (1,10):
for hyperparam in [1]:
    look_back = 10 * hyperparam
    # look_back = 10
    # add historical volatility, and ma
    n = len(percentized)
    for i in range(0, n):
        std = 0.0
        ma = 0.0
        # less than look_back
        if i == 0:
            std = 0.0
            ma = 0.0
        elif i < look_back:
            std = np.log(np.std(raw[0:i, 3]))
            std = np.min(std, -1)
            ma = np.average(raw[0:i, 3])
            ma = np.log(raw[i, 3]/ma)
        else:
            std = np.log(np.std(raw[i-look_back:i, 3]))
            ma = np.average(raw[i-look_back:i, 3])
            ma = np.log(raw[i, 3] / ma)
        hv = std / 3
        hv = min(hv, 1)
        hv = max(hv, -1)
        percentized[i, 5] = hv
        percentized[i, 6] = ma

    # make it predictive
    labels = labels[1:]
    percentized = percentized[:-1]

    # remove the first 31 rows that don't have the correct std because there wasn't enough historical data
    labels = labels[look_back - 1:]
    percentized = percentized[look_back -1:]

    # batch it
    time_steps = 2 ** 5
    data_samples = []
    label_samples = []
    num_time_pts = len(labels)
    for i in range(0, num_time_pts):
        sample_data = []
        for j in range(0, time_steps):
            if i + j >= num_time_pts:
                break
            sample_data.append(percentized[i + j])
        if len(sample_data) == time_steps:
            data_samples.append(sample_data)
            label_samples.append(labels[i + j])

    data_samples = np.asarray(data_samples, dtype='float16')
    label_samples = np.asarray(label_samples, dtype='float16')

    if save_data:
        np.savez("processedX.npz", data_samples)
        np.savez("processedY.npz", label_samples)

    print('xy hyperparam', hyperparam)
    optAndNotify(data_samples, label_samples)
