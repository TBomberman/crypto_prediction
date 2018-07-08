import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.callbacks import History, EarlyStopping
from random import sample
from utilities import minmax, remove_constant_values, all_stats
# import matplotlib.pyplot as plt
import keras_enums as enums
import random
from sklearn.model_selection import train_test_split
from helpers.callbacks import NEpochLogger

# local variables
dropout = 0.0
batch_size = 2**11
nb_epoch = 10000
train_percentage = 0.7
hidden_layer_count = 1
patience = 10

# for reproducibility
# np.random.seed(1337)
# random.seed(1337)

def do_optimize(nb_classes, data, labels, data_test=None, labels_test=None):
    n = len(labels)
    time_steps = len(data[0])
    d = data[0][0].size
    neuron_count = time_steps * d
    if nb_classes:
        labels = np_utils.to_categorical(labels, nb_classes)
    train_size = int(train_percentage * n)
    print("Train size:", train_size)
    test_size = int((1 - train_percentage) * n)

    # predict the last section
    # indexes = np.arange(start=n - test_size, stop=n)
    # np.random.shuffle(indexes)
    # X_test = data[indexes]
    # y_test = labels[indexes]
    # indexes = np.arange(n-test_size)
    # np.random.shuffle(indexes)
    # X_train = data[indexes]
    # y_train = labels[indexes]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, test_size=test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5)

    # for hyperparam in range(0, 10):
    for hyperparam in [1]:
        # 0: 'sgd', 1: 'rmsprop', 2: 'adagrad', 3: 'adadelta', 4: 'adam', 5: 'adamax', 6: 'nadam'
        optimizer = enums.optimizers[6] #rmsprop
        # 0: 'elu', 1: 'selu', 2: 'sigmoid', 3: 'linear', 4: 'softplus', 5: 'softmax', 6: 'tanh',
        # 7: 'hard_sigmoid', 8: 'relu', 9: 'softsign'
        activation_input = enums.activation_functions[8]
        activation_hidden = enums.activation_functions[6]
        activation_output = enums.activation_functions[5]

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
        print('Patience', patience)
        out_epoch = NEpochLogger(display=5)

        model = Sequential()
        history = History()
        model.add(LSTM(neuron_count, input_shape=(time_steps, d), return_sequences=True))
        model.add(Activation(activation_input))
        model.add(Dropout(dropout))

        add_lstm_dropout(hidden_layer_count, neuron_count, model, activation_hidden)

        model.add(Dense(nb_classes))
        model.add(Activation(activation_output))
        # model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=0, validation_data=(X_test, y_test), callbacks=[history, early_stopping, out_epoch])

        score = model.evaluate(X_test, y_test, verbose=0)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        y_score_train = model.predict_proba(X_train)
        y_score_test = model.predict_proba(X_test)
        y_score_val = model.predict_proba(X_val)

        print("up")
        for i in range(0, 10):
            threshold = 0.5 + 0.05*i
            locations = np.where(y_score_test[:, 1] >= threshold)
            n = len(locations[0])
            if n <= 0:
                continue
            sum = np.sum(y_test[locations, 1])
            acc = sum / n
            print('Percentile', str(threshold), 'cross threshold', n, 'matching', sum, 'My Test accuracy:', acc)

        print("down")
        for i in range(0, 10):
            threshold = 0.5 + 0.05*i
            locations = np.where(y_score_test[:, 0] > threshold)
            n = len(locations[0])
            if n <= 0:
                continue
            sum = np.sum(y_test[locations, 0])
            acc = sum / n
            print('Percentile', str(threshold), 'cross threshold', n, 'matching', sum, 'My Test accuracy:', acc)
        # y_score_val[np.where(y_score_val >= 0.5)] = 1
        # y_score_val[np.where(y_score_val < 0.5)] = 0
        # np.savetxt("predictions_val.csv", y_score_val[:, 1], delimiter=",")
        # np.savetxt("predictions_test.csv", y_score_test[:, 1], delimiter=",")
        # np.savetxt("predictions_train.csv", y_score_train[:, 1], delimiter=",")

        if nb_classes > 1:
            train_stats = all_stats(y_train[:, 1], y_score_train[:, 1])
            val_stats = all_stats(y_val[:, 1], y_score_val[:, 1])
            test_stats = all_stats(y_test[:, 1], y_score_test[:, 1], val_stats[-1])
        else:
            train_stats = all_stats(y_train, y_score_train)
            test_stats = all_stats(y_test, y_score_test, train_stats[-1])
            val_stats = all_stats(y_val, y_score_val, train_stats[-1])

        print_out = 'Hidden layers: %s, Neurons per layer: %s, Hyperparam: %s' % (hidden_layer_count + 1, neuron_count, hyperparam)
        print(print_out)
        print('All stats train:', ['{:6.2f}'.format(val) for val in train_stats])
        print('All stats test:', ['{:6.2f}'.format(val) for val in test_stats])
        print('All stats val:', ['{:6.2f}'.format(val) for val in val_stats])
        # print(history.history.keys())
        # summarize history for loss

        # plot
        # nth = int(nb_epoch *0.05)
        # nth = 1
        # five_ploss = history.history['loss'][0::nth]
        # five_pvloss = history.history['val_loss'][0::nth]
        # plt.figure()
        # plt.plot(five_ploss)
        # plt.plot(five_pvloss)
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.draw()
        #
        # plt.show()

def add_lstm_dropout(count, neuron_count, model, activation):
    for x in range(0, count):
        if x == count - 1:
            model.add(LSTM(neuron_count, return_sequences=False))
        else:
            model.add(LSTM(neuron_count, return_sequences=True))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
