from helpers import data_loader as dl
import numpy as np
from lstm_optimizer import do_optimize
from helpers.email_notifier import notify

def standardize(samples):
    standardized = np.zeros(samples.shape)
    for sample_idx in range(0, len(samples)):
        sample = samples[sample_idx]
        # sample has time steps and features
        for feature_idx in range(0, sample.shape[1]):
            feature = sample[:, feature_idx]
            mean = np.mean(feature)
            std = np.std(feature)
            for value_idx in range(0, samples.shape[1]):
                standardized_value = (feature[value_idx] - mean) / std
                standardized[sample_idx, value_idx, feature_idx] = standardized_value
    return standardized

def preprocess(samples):
    processed = np.zeros(samples.shape)
    for sample_idx in range(0, len(samples)):
        # standardize the prices
        sample = samples[sample_idx]
        # sample has time steps and features
        for feature_idx in range(0, sample.shape[1]-1):
            feature = sample[:, feature_idx]
            mean = np.mean(feature)
            std = np.std(feature)
            for value_idx in range(0, samples.shape[1]):
                standardized_value = (feature[value_idx] - mean) / std
                processed[sample_idx, value_idx, feature_idx] = standardized_value

        #normalize the
        vol_feature_idx = sample.shape[1]
        vol_feature = sample[:, feature_idx]
        min = vol
    return processed

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
        pct_row.append(float(row[i])/float(last_row[i])-1)
    pct_row.append(float(row[6]))
    percentized.append(pct_row)
    if row[5] > last_row[5]:
        labels.append(1)
    else:
        labels.append(0)
    last_row = row
labels.pop(0)

# np it
percentized = np.asarray(percentized)

# standardize the percents
for i in range(0,4):
    mean = np.mean(percentized[:,i])
    std = np.std(percentized[:,i])
    percentized[:, i] = (percentized[:,i] - mean) / std

# normalize volume
min_vol = np.min(percentized[:,4])
max_vol = np.max(percentized[:,4])
percentized[:,4] = (percentized[:,4] - min_vol)/(max_vol - min_vol)

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

try:
    data_samples = np.asarray(data_samples, dtype='float16')
    label_samples = np.asarray(label_samples, dtype='float16')
    do_optimize(2, data_samples, label_samples)
finally:
    notify("cccy")
