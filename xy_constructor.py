from helpers import data_loader as dl
import numpy as np
from lstm_optimizer import do_optimize
from helpers.email_notifier import notify

save_data = False
load_data = False

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
headers = csv_rows.pop(0)

percentized = []
last_row = None
labels = []
# %ize the prices
for row in reversed(csv_rows): # oldest to newest
    if last_row == None:
        last_row = row
        continue
    pct_row = []
    for i in range(2, 6):
        pct_row.append(float(row[i])/float(last_row[i]))
    pct_row.append(float(row[6]))
    pct_row.append(0.0)
    percentized.append(pct_row)
    if row[5] > last_row[5]:
        labels.append(1)
    else:
        labels.append(0)
    last_row = row
labels.pop(0)

# np it
percentized = np.log(np.asarray(percentized))

# standardize the percents
for i in range(0,4):
    mean = np.mean(percentized[:,i])
    std = np.std(percentized[:,i])
    percentized[:, i] = (percentized[:,i] - mean) / std

# standardize the volume
mean = np.mean(percentized[:,4])
std = np.std(percentized[:,4])
percentized[:, 4] = (percentized[:,4] - mean) / std

# add historical volatility
n = len(percentized)
for i in range(0, n):
    std = 0.0
    # less than 32
    if i == 0:
        std = 0.0
    elif i < 32:
        std = np.log(np.std(percentized[0:i,3]))
    else:
        std = np.log(np.std(percentized[i-32:i, 3]))
    hv = std / 3
    hv = min(hv, 1)
    hv = max(hv, -1)
    percentized[i, 5] = hv


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

optAndNotify(data_samples, label_samples)
