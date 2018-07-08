from helpers import data_loader as dl
import numpy as np
# from lstm_optimizer import do_optimize
from ensemble_optimizer import do_optimize
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

csv_rows = dl.load_csv('Data/Candles15mBTC-ETH-LTCJul2018.csv')[60000:]
# csv_rows = dl.load_csv('Data/Candles1HDec2014-May2018.csv')
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
    for i in range(2, 6):
        pct_row.append(float(row[i])/float(last_row[i]))
        raw_row.append(float(row[i]))
    pct_row.append(float(row[6]))
    for i in range(7, 11):
        pct_row.append(float(row[i])/float(last_row[i]))
        raw_row.append(float(row[i]))
    pct_row.append(float(row[11]))
    for i in range(12, 16):
        pct_row.append(float(row[i])/float(last_row[i]))
        raw_row.append(float(row[i]))
    pct_row.append(float(row[16]))
    pct_row.append(0.0)  # for hv
    pct_row.append(0.0)  # for hv
    pct_row.append(0.0)  # for hv
    pct_row.append(0.0)  # for ma
    pct_row.append(0.0)  # for ma
    pct_row.append(0.0)  # for ma
    pct_row.append(0.0)  # for macd
    pct_row.append(0.0)  # for macd
    pct_row.append(0.0)  # for macd
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
    return pctmean, pctstd

pctmean1, pctstd1 = standardize_percents(0, 4)
pctmean2, pctstd2 = standardize_percents(5, 9)
pctmean3, pctstd3 = standardize_percents(10, 14)

def standardize_log_vol(vol_col):
    # standardize the log volume
    percentized[:,vol_col] = np.log(percentized[:,vol_col])
    mean = np.mean(percentized[:,vol_col])
    std = np.std(percentized[:,vol_col])
    percentized[:, vol_col] = (percentized[:,vol_col] - mean) / (std*3)
standardize_log_vol(4)
standardize_log_vol(9)
standardize_log_vol(14)

# for hyperparam in range (1,10):
for hyperparam in [1]:
    look_back = 10 * hyperparam
    # look_back = 10
    # add historical volatility, and ma
    n = len(percentized)
    for i in range(0, n):
        std = 0.0
        logstd1 = 0.0
        logstd2 = 0.0
        logstd3 = 0.0
        ma1 = 0.0
        ma2 = 0.0
        ma3 = 0.0
        # less than look_back
        if i == 0:
            ma = 0.0
        elif i < look_back:
            def get_ls_ma(close_col):
                std = np.std(raw[0:i, close_col])
                if std == 0:
                    logstd = -3
                else:
                    logstd = np.log(std)
                ma = np.average(raw[0:i, close_col])
                ma = (raw[i, close_col]/ma - pctmean1) / (pctstd1*3)
                return logstd, ma
            logstd1, ma1 = get_ls_ma(3)
            logstd2, ma2 = get_ls_ma(7)
            logstd3, ma3 = get_ls_ma(11)
        else:
            def get_ls_ma2(close_col):
                std = np.std(raw[i-look_back:i, close_col])
                if std == 0:
                    logstd = -3
                else:
                    logstd = np.log(std)
                ma = np.average(raw[i-look_back:i, close_col])
                ma = (raw[i, close_col]/ma - pctmean1) / (pctstd1*3)
                return logstd, ma
            logstd1, ma1 = get_ls_ma2(3)
            logstd2, ma2 = get_ls_ma2(7)
            logstd3, ma3 = get_ls_ma2(11)
        def get_hv(logstd):
            hv = logstd / 3
            hv = min(hv, 1)
            hv = max(hv, -1)
            return hv
        hv1 = get_hv(logstd1)
        hv2 = get_hv(logstd2)
        hv3 = get_hv(logstd3)
        percentized[i, 15] = hv1
        percentized[i, 16] = hv2
        percentized[i, 17] = hv3
        percentized[i, 18] = ma1
        percentized[i, 19] = ma2
        percentized[i, 20] = ma3

    look_back12 = 12
    look_back26 = 26

    # add macd
    def get_macd(close_col):
        n = len(percentized)
        working = np.zeros(n)
        for i in range(0, n):
            if i < look_back12:
                ma12 = np.average(raw[0:i, close_col])
            if i < look_back26:
                ma26 = np.average(raw[0:i, close_col])
            if i >= look_back12:
                ma12 = np.average(raw[i - look_back12:i, close_col])
            if i >= look_back26:
                ma26 = np.average(raw[i - look_back26:i, close_col])
            if i == 0:
                ma12 = 0.0
                ma26 = 0.0
            working[i] = ma12 - ma26
        mean = np.mean(working)
        std = np.std(working)
        return [(x - mean) / (std*3) for x in percentized[:, 7]]

    percentized[:, 21] = get_macd(3)
    percentized[:, 22] = get_macd(7)
    percentized[:, 23] = get_macd(11)

    # make it predictive
    labels = labels[1:]
    percentized = percentized[:-1]

    # remove the first 26 rows that don't have the correct std because there wasn't enough historical data
    labels = labels[look_back26 - 1:]
    percentized = percentized[look_back26 - 1:]

    for i in range(0, 24):
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
