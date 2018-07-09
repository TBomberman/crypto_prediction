from helpers import data_loader as dl
import numpy as np
from lstm_optimizer import do_optimize
# from ensemble_optimizer import do_optimize
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

csv_rows = dl.load_csv('Data/Candles15mBTC-ETH-LTCJul2018.csv')
headers = csv_rows.pop(0)

percentized = []
raw = []
last_row = None
labels = []
# %ize the prices
for row in csv_rows: # oldest to newest
    if last_row == None:
        last_row = row
        continue
    pct_row = []
    raw_row = []
    for i in range(12, 16):
        pct_row.append(float(row[i])/float(last_row[i]))
        raw_row.append(float(row[i]))
    pct_row.append(float(row[16]))
    percentized.append(pct_row)

    # create labels
    if row[15] > last_row[15]:
        labels.append(1)
    else:
        labels.append(0)
    last_row = row

    # save raw and working
    raw.append(raw_row)

# np them
percentized = np.asarray(percentized)
# percentized = np.min(percentized, -1)
labels = np.asarray(labels)
raw = np.asarray(raw)

def standardize_percents(start, end):
    # standardize the percents
    pctmean = np.mean(percentized[:, start:end])
    pctstd = np.std(percentized[:, start:end])
    percentized[:, start:end] = (percentized[:, start:end] - pctmean) / (pctstd*3)
standardize_percents(0, 4)

def standardize_log_vol(vol_col):
    # standardize the log volume
    percentized[:,vol_col] = np.log(percentized[:,vol_col])
    mean = np.mean(percentized[:,vol_col])
    std = np.std(percentized[:,vol_col])
    percentized[:, vol_col] = (percentized[:,vol_col] - mean) / (std*3)
standardize_log_vol(4)

# for hyperparam in range (1,10):
for hyperparam in [1]:
    # make it predictive
    labels = labels[1:]
    percentized = percentized[:-1]

    for i in range(0, len(percentized[0])):
        print('stds', np.std(percentized[:, i]))

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
