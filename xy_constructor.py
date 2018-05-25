from helpers import data_loader as dl
import numpy as np
from lstm_optimizer import do_optimize
from helpers.email_notifier import notify

csv_rows = dl.load_csv('Data/CandlesJan2015-Dec2017.csv')
headers = csv_rows.pop(0)
time_steps = 10

# %ize it
percentized = []
last_row = None
time_series_labels = []
for row in reversed(csv_rows):
    if last_row == None:
        last_row = row
        continue
    pct_row = []
    for i in range(2, 7):
        pct_row.append(float(row[i])/float(last_row[i])-1)
    percentized.append(pct_row)
    if row[5] > last_row[5]:
        time_series_labels.append(1)
    else:
        time_series_labels.append(0)
    last_row = row
time_series_labels.pop(0)
percentized.pop(len(percentized)-1)
percentized = np.asarray(percentized, dtype=np.float16)

# normalize it
time_series_data = np.tanh(percentized)

try:
    for hyperparam in range(4, 6):
        # batch it
        time_steps = 2 ** hyperparam
        data_samples = []
        label_samples = []
        num_samples = len(time_series_data)
        for i in range(0, num_samples):
            sample_data = []
            for j in range(0, time_steps):
                if i + j >= num_samples:
                    break
                sample_data.append(time_series_data[i+j])
            if len(sample_data) == time_steps:
                data_samples.append(sample_data)
                label_samples.append(time_series_labels[i+j])

        data_samples = np.asarray(data_samples, dtype='float16')
        label_samples = np.asarray(label_samples, dtype='float16')
        print("xy hyperparam:", hyperparam)
        do_optimize(2, data_samples, label_samples)
finally:
    notify()