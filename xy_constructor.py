from helpers import data_loader as dl
import numpy as np
from lstm_optimizer import do_optimize

csv_rows = dl.load_csv('Data/CandlesJan2015-Dec2017.csv')
headers = csv_rows.pop(0)

# %ize it
percentized = []
last_row = None
labels = []
for row in reversed(csv_rows):
    if last_row == None:
        last_row = row
        continue
    pct_row = []
    for i in range(2, 7):
        pct_row.append(float(row[i])/float(last_row[i])-1)
    percentized.append(pct_row)
    if row[5] > last_row[5]:
        labels.append(1)
    else:
        labels.append(0)
    last_row = row
labels.pop(0)
percentized.pop(len(percentized)-1)
percentized = np.asarray(percentized, dtype=np.float16)

# normalize it
data = np.tanh(percentized)

do_optimize(2, data, labels)
