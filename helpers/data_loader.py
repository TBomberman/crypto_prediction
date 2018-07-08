import numpy as np
import csv

# local vars
cutoff = 0.5

def ensure_number(data):
    return np.nan_to_num(data)

def load_csv(file):
        #load data
        expression = []
        with open(file, "r") as csv_file:
            reader = csv.reader(csv_file, dialect='excel')
            for row in reader:
                expression.append(row)

        return expression

def load_descriptors(file):
        descriptors = []
        with open(file, "r") as tab_file:
            reader = csv.reader(tab_file, dialect='excel', delimiter='\t')
            descriptors = dict((rows[1],rows[2:]) for rows in reader)

        print('drug descriptors loaded. rows:  ' + str(len(descriptors)))
        return descriptors

def join_descriptors_label(expression,descriptors):
        unique_drugs = []
        # data set up
        data = []
        for row in expression:
            data.append(descriptors[row[0]])
            if row[0] not in unique_drugs:
                unique_drugs.append(row[0])
        data = np.array(data).astype(np.float32)

        labels = []
        for row in expression:
            labels.append(row[1:3])

        labels = np.array(labels).astype(np.float32)
        print('data size ' + str(len(data)) + ' labels size ' + str(len(labels)))
        return data,labels

def get_feature_dict(file, delimiter=',', key_index=0, use_int=False):
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=delimiter)
        next(reader)
        if use_int:
            my_dict = {}
            for row in reader:
                list = []
                for value in row[1:]:
                    list.append(int(value))
                my_dict[row[key_index]] = list
            return my_dict
        return dict((row[key_index], row) for row in reader)

def printProgressBar (iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 25, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')
    # Print New Line on Complete
    if iteration == total:
        print()