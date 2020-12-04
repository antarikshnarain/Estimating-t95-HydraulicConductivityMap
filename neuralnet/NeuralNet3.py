"""
Author: Antariksh Narain
Description: Setup a neural network to train data
Design:
-> Dropout 0.25, 0.5
-> Check Train and Eval flag in predict
-> conv layer 3 and 5.
-> data [0.6, 0.2, 0.2]
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from Dataset import Dataset
from VisualizeMap import Visualize

import keras
import tensorflow as tf
import numpy as np

from scipy.stats import pearsonr

import csv
import os

from datetime import datetime

MODEL_PATH = "weights/"
TRAINED_PATH = "model/"
PERF_PATH = "performance/"

class NeuralNet:
    def __init__(self, split=[70,20,10], filename="model"):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        if not os.path.exists(TRAINED_PATH):
            os.makedirs(TRAINED_PATH)
        if not os.path.exists(PERF_PATH):
            os.makedirs(PERF_PATH)
        
        self.viz = Visualize()
        self.dataset = Dataset()
        self.dataset.LoadData("Field_Serra*.csv", "list_t*.csv", split)
        self.filename = filename

    def _processX(self, X, count):
        new_X = []
        for _ in range(count):
            new_X.append(X[:,_].reshape(100,205))
        return np.array(new_X)
    
    def _get_data(self, data_type):
        x, y, size = self.dataset.MiniBatch(data_type)
        x = self._processX(x, size)
        x = x.reshape((size, 100, 205, 1))
        y = y.reshape((size,))
        return x,y

    def _accuracy(self, lst_X: list(), lst_Y: list()):
        assert(len(lst_X) == len(lst_Y))
        accuracy = [self.model.evaluate(x, y) for x,y in zip(lst_X, lst_Y)]
        return accuracy

    def _prediction(self, lst_X: list, lst_Y: list):
        assert(len(lst_X) == len(lst_Y))
        y_predic = [self.model.predict(x).reshape(y.shape[0],) for x,y in zip(lst_X, lst_Y)]
        return y_predic

    def _check_predictions(self, X, Y):
        y_p = self.model.predict(X)
        y_p = y_p.reshape(y_p.shape[0],)
        corr, _ = pearsonr(Y, y_p.reshape(y_p.shape[0],))
        print("Pearson Score: %.3f"%corr)
        diff = y_p - Y
        ct_p = 0
        ct_n = 0
        ct_0 = 0
        for a in diff:
            if a > 0:
                ct_p+=1
            elif a < 0:
                ct_n+=1
            else:
                ct_0+=1
        return ct_p, ct_n, ct_0, corr
    
    def _write_to_file(self, rows):
        f = open(PERF_PATH + self.filename + ".csv", "a")
        cc = csv.writer(f)
        cc.writerows(rows)
        f.close()
        print("Performance data written to CSV")

    def BuildModel(self, loss_type='mean_squared_error', dropout=0.0):
        self.model = Sequential()

        self.model.add(Conv2D(16, kernel_size=7, padding='same', strides=(2,2), activation='relu', input_shape=(100,205,1)))
        #self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=2))

        # self.model.add(Conv2D(32, kernel_size=7, padding='same', activation='relu'))
        # #self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(strides=2))
        
        self.model.add(Conv2D(64, kernel_size=7, padding='same', activation='relu'))
        #self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=2))
        
        # self.model.add(Conv2D(128, kernel_size=7, padding='same', activation='relu'))
        # #self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(strides=2))

        self.model.add(Conv2D(256, kernel_size=5, padding='same', activation='relu'))
        #self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=2))
        
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        #self.model.compile(loss='mean_squared_error', optimizer='adam')
        #self.model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
        self.model.compile(loss=loss_type, optimizer='adam')

        self.model.summary()
        plot_model(self.model, to_file="model.png")
   
    
    def TrainModel(self, epochs, batch_size=10, max_iteration = 20):
        x_train, y_train = self._get_data('train')
        x_val, y_val = self._get_data('val')
        x_test, y_test = self._get_data('test')
        
        checkpoint = ModelCheckpoint(MODEL_PATH + self.filename + ".hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=5)
        old_predic = 0
        ctr = 0
        eps = int(epochs / max_iteration)
        self._write_to_file([["Train","Validation", "Test"]])
        for itr in range(max_iteration):
            print("Running Iteration ", itr)
            self.model.fit(x_train, y_train, epochs=eps, batch_size=batch_size, verbose=0, validation_data=(x_val, y_val), callbacks=checkpoint)
        
            print("\n\nEvaluating Models")
            accuracy = self._accuracy([x_train, x_val, x_test], [y_train, y_val, y_test])
            print("Accuracy: Train->%.2f | Val->%.2f | Test->%.2f\n"%(accuracy[0], accuracy[1], accuracy[2]))
            print("\n\nPredictions")
            prediction = self._prediction([x_train, x_val, x_test], [y_train, y_val, y_test])
            predic1 = self._check_predictions(x_train, y_train)
            predic2 = self._check_predictions(x_val, y_val)
            predic3 = self._check_predictions(x_test, y_test)

            self._write_to_file([accuracy, [predic1, predic2, predic3]])

            ## Early Stopping
            if old_predic < predic2[3]:
                old_predic = predic2[3]
                ctr=0
                self.save()
            else:
                if ctr == 3:
                    print("Breaking Early!", old_predic, predic2[3])
                    break
                ctr += 1
                #old_predic = predic2[3]
        self._write_to_file([["Index", "Actual","Predicted"]])
        # self._write_to_file(np.transpose([np.array(y_val.tolist()+y_test.tolist()), 
        #     prediction[1].tolist() + prediction[2].tolist()]))
        self._write_to_file(np.transpose([
            np.array(self.dataset.Val_seq + self.dataset.Test_seq),
            np.array(y_val.tolist()+y_test.tolist()), 
            np.array(prediction[1].tolist() + prediction[2].tolist())]))
        return [predic1, predic2, predic3]

    def save(self):
        self.model.save(TRAINED_PATH + filename + ".hdf5")
        print("Saved Model!")


if __name__ == "__main__":
    #dataset_perm = [[60,20,20],[70,20,10],[80,10,10],[85,10,5]]
    batch_size = [5,20,50]
    dataset_perm = [[60,20,20]]
    loss_types = ['mean_squared_error', 'mean_absolute_percentage_error']
    #loss_types = ['mean_absolute_percentage_error']
    dropouts = [0.25, 0.5]
    for dp in dataset_perm:
        for batchsize in batch_size:
            for loss_type in loss_types:
                for dropout in dropouts:
                    filename = str(datetime.now()) + "_Batch_" + str(batchsize) + "_Loss_" + str(loss_type) + "_D_" + str(dropout)
                    nn = NeuralNet(dp, filename)
                    nn.BuildModel(loss_type=loss_types, dropout=dropout)
                    results = nn.TrainModel(epochs=200,batch_size=batchsize,max_iteration=20)
                    print("=====Performance=====")
                    print("Batch ", batch_size, " Loss ", loss_type, " Dropout ", dropout)
                    print(results)
                    print("---------------------")
                    nn.save()
                    del nn
