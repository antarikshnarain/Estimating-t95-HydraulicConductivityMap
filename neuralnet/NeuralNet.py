"""
Author: Antariksh Narain
Description: Setup a neural network to train data
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from Dataset import Dataset
from VisualizeMap import Visualize

import keras
import tensorflow as tf
import numpy as np

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
        diff = y_p.reshape(y_p.shape[0],) - Y
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
        return ct_p, ct_n, ct_0
    
    def _write_to_file(self, rows):
        f = open(PERF_PATH + self.filename + ".csv", "a")
        cc = csv.writer(f)
        cc.writerows(rows)
        f.close()
        print("Performance data written to CSV")

    def BuildModel(self):
        self.model = Sequential()

        # self.model.add(Conv2D(16, kernel_size=7, padding='same', strides=(2,2), activation='relu', input_shape=(100,205,1)))
        # self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(strides=2))

        # self.model.add(Conv2D(32, kernel_size=7, padding='same', activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(strides=2))
        
        # self.model.add(Conv2D(64, kernel_size=7, padding='same', activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(strides=2))
        
        # self.model.add(Conv2D(128, kernel_size=7, padding='same', activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(strides=2))

        # self.model.add(Conv2D(256, kernel_size=5, padding='same', activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(strides=2))

        self.model.add(Conv2D(128, kernel_size=7, padding='same', strides=(2,2), activation='relu', input_shape=(100,205,1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=2))
        self.model.add(Conv2D(256, kernel_size=5, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=2))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        # self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(1, activation='relu'))

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary()
        plot_model(self.model, to_file="model.png")
   
    
    def TrainModel(self, epochs, batch_size=10):
        x_train, y_train = self._get_data('train')
        x_val, y_val = self._get_data('val')
        x_test, y_test = self._get_data('test')
        # self.viz.plot(x_train[1197,:,:,:])
        # self.viz.save("map_1197.png")
        
        checkpoint = ModelCheckpoint(MODEL_PATH + self.filename + ".hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=5)

        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val), callbacks=checkpoint)
        
        print("\n\nEvaluating Models")
        accuracy = self._accuracy([x_train, x_val, x_test], [y_train, y_val, y_test])
        print("Accuracy: Train->%.2f | Val->%.2f | Test->%.2f\n"%(accuracy[0], accuracy[1], accuracy[2]))
        print("\n\nPredictions")
        prediction = self._prediction([x_train, x_val, x_test], [y_train, y_val, y_test])
        predic1 = self._check_predictions(x_train, y_train)
        predic2 = self._check_predictions(x_val, y_val)
        predic3 = self._check_predictions(x_test, y_test)
        predic_str = "Train: " + str(predic1) + "\tVal: " +  str(predic2)  + "\tTest: " + str(predic3) + "\n"
        print(predic_str)

        self._write_to_file([["Train","Validation", "Test"],accuracy, [predic1, predic2, predic3], ["Index","Actual", "Predicted"]])
        # self._write_to_file(np.transpose([np.array(y_train.tolist()+y_val.tolist()+y_test.tolist()), 
        #     np.array(prediction[0].tolist() + prediction[1].tolist() + prediction[2].tolist())]))
        self._write_to_file(np.transpose([
            np.array(self.dataset.Train_seq + self.dataset.Val_seq + self.dataset.Test_seq),
            np.array(y_train.tolist()+y_val.tolist()+y_test.tolist()), 
            np.array(prediction[0].tolist() + prediction[1].tolist() + prediction[2].tolist())]))

    def save(self):
        self.model.save(TRAINED_PATH + filename + ".hdf5")
        print("Saved Model!")


if __name__ == "__main__":
    #dataset_perm = [[60,20,20],[70,20,10],[80,10,10],[85,10,5]]
    #batch_size = [5,10,20,50]
    dataset_perm = [[60,20,20]]
    batch_size = [20]
    for dp in dataset_perm:
        for batchsize in batch_size:
            filename = str(datetime.now()) + "_".join([str(t) for t in dp]) + "_Batch_" + str(batchsize)
            nn = NeuralNet(dp, filename)
            nn.BuildModel()
            nn.TrainModel(epochs=1,batch_size=batchsize)
            nn.save()
            del nn