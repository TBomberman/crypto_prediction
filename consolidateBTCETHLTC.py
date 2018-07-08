from helpers import data_loader as dl
import csv

btc_dict = dl.get_feature_dict('Data/Candles15mBTC-ETH-LTCJul2018.csv', key_index=1)
eth_dict = dl.get_feature_dict('Data/ETH.csv', key_index=1)
ltc_dict = dl.get_feature_dict('Data/LTC.csv', key_index=1)

start = 1471975200
end = 1530985500

csv_rows = []
current = start
btc_row = []
eth_row = []
ltc_row = []
while current <= end:
    date = ''
    current_str = str(current)
    if current_str in btc_dict:
        btc_row = btc_dict[current_str]
        date = btc_row[0]
    if current_str in eth_dict:
        eth_row = eth_dict[current_str]
        date = eth_row[0]
    if current_str in ltc_dict:
        ltc_row = ltc_dict[current_str]
        date = ltc_row[0]
    csv_rows.append([date, current_str] + btc_row[2:] + eth_row[2:] + ltc_row[2:])
    print(current)
    current += 900

with open('temp.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

