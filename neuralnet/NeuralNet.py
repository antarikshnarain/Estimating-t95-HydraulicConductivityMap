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
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from Dataset import Dataset
from VisualizeMap import Visualize, VisualizeLayers
from Utils import Utility

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
LAYER_IMG = "layers/"

class NeuralNet:
    def __init__(self, split=[70,20,10], filename="model"):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        if not os.path.exists(TRAINED_PATH):
            os.makedirs(TRAINED_PATH)
        if not os.path.exists(PERF_PATH):
            os.makedirs(PERF_PATH)
        if not os.path.exists(LAYER_IMG):
            os.makedirs(LAYER_IMG)

        self.viz = Visualize()
        self.dataset = Dataset()
        self.utility = Utility(filename)
        self.dataset.LoadData("Field_Serra*.csv", "list_t*.csv", split)
        self.filename = filename

    def _processX(self, X, count):
        new_X = []
        for _ in range(count):
            new_X.append(np.flip(X[:,_].reshape(100,205),axis=0))
        return np.array(new_X)
    
    def _get_data(self, data_type):
        x, y, size = self.dataset.MiniBatch(data_type)
        x = self._processX(x, size)
        x = x.reshape((size, 100, 205, 1))
        y = y.reshape((size,))
        return x,y
    
    def _get_specific_records(self, lst:list):
        x, y, size = self.dataset.CustomBatch(lst)
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
        # y_p = self.model.predict(lst_X)
        # y_p = y_p.reshape(y_p.shape[0],)
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

    def BuildModel(self, layers: list, loss_type='mean_squared_error', dropout=0.0):
        self.model = Sequential()

        for layer in layers:
            self.model.add(layer)

        self.model.compile(loss=loss_type, optimizer='adam')#, metrics=['accuracy'])
        self.model.summary()
        plot_model(self.model, to_file=self.filename + "_model.png")   
    
    def TrainModel(self, epochs, batch_size=10):
        x_train, y_train = self._get_data('train')
        x_val, y_val = self._get_data('val')
        x_test, y_test = self._get_data('test')
        
        #checkpoint = ModelCheckpoint(MODEL_PATH + self.filename + ".hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=10)
        checkpoint = ModelCheckpoint(MODEL_PATH + self.filename + ".hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=10)
        self._write_to_file([["Train","Validation", "Test"]])

        #history = self.model.fit(x_train, y_train, steps_per_epoch=int(x_train.shape[0]/epochs), epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_val, y_val), callbacks=checkpoint)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_val, y_val), callbacks=checkpoint)

        print("\n\nEvaluating Models")
        accuracy = self._accuracy([x_train, x_val, x_test], [y_train, y_val, y_test])
        print(accuracy)

        print("\n\nPredictions")
        prediction = self._prediction([x_train, x_val, x_test], [y_train, y_val, y_test])
        predic1 = self._check_predictions(x_train, y_train)
        predic2 = self._check_predictions(x_val, y_val)
        predic3 = self._check_predictions(x_test, y_test)

        self.X = x_test

        # acc = history.history['accuracy']
        # val_acc = history.history['val_accuracy']
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # tot_epochs = range(1, len(acc) + 1)
        #self.utility.display_accuracy(tot_epochs, acc, val_acc)
        #self.utility.display_loss(tot_epochs, loss, val_loss)
        #self.utility.show_layers(self.model, x_test)
        self._write_to_file([accuracy, [predic1, predic2, predic3]])

        self._write_to_file([["Index", "Actual","Predicted"]])
        # self._write_to_file(np.transpose([np.array(y_val.tolist()+y_test.tolist()), 
        #     prediction[1].tolist() + prediction[2].tolist()]))
        self._write_to_file(np.transpose([
            np.array(self.dataset.Val_seq + self.dataset.Test_seq),
            np.array(y_val.tolist()+y_test.tolist()), 
            np.array(prediction[1].tolist() + prediction[2].tolist())]))
        return [predic1, predic2, predic3]

    def save(self):
        self.model.save(TRAINED_PATH + self.filename + ".hdf5")
        print("Saved Model!")

    def load(self):
        self.model.load_weights(MODEL_PATH + self.filename + ".hdf5")
    
    def visualize(self, data_indexs: list, save=False):
        vl = VisualizeLayers()
        for index in data_indexs:
            datas,_ = self._get_specific_records([index])
            if not os.path.exists(LAYER_IMG + str(index)):
                os.makedirs(LAYER_IMG + str(index))
            vl.visualize(self.model, datas, save_image=save, path=LAYER_IMG + str(index))
    
    def predict(self, data_indexs: list):
        X,Y = self._get_specific_records(data_indexs)
        y_p = self.model.predict(X)
        print(self._prediction(X, Y))
