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

import time

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)
# tf.logging.set_verbosity(tf.logging.ERROR)

class NeuralNet:
    def __init__(self, split=[70,20,10]):
        self.viz = Visualize()
        self.dataset = Dataset()
        self.dataset.LoadData("dataset/Field_Serra*.csv", "dataset/list_t*.csv", split)

    def build_model(self, input_dim, output_dim):
        self.model = Sequential()

        #self.model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
        self.model.add(Conv2D(16, kernel_size=7, padding='same', strides=(2,2), activation='relu', input_shape=(100,205,1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=2))

        self.model.add(Conv2D(32, kernel_size=7, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=2))
        
        self.model.add(Conv2D(64, kernel_size=7, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=2))
        
        self.model.add(Conv2D(128, kernel_size=7, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=2))

        self.model.add(Conv2D(256, kernel_size=5, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=2))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        #self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(1, activation='relu'))

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary()
        plot_model(self.model, to_file="model.png")
   
    
    def train_model(self, epochs, batch_size=10, model_filename="best_model.hdf5"):
        x_train, y_train = self.GetData('train')
        x_val, y_val = self.GetData('val')
        x_test, y_test = self.GetData('test')
        self.viz.plot(x_train[1197,:,:,:])
        self.viz.save("map_1197.png")
        
        checkpoint = ModelCheckpoint(model_filename, monitor='loss', verbose=1, save_best_only=True, mode='auto', period=5)

        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val), callbacks=checkpoint)
        
        print("\n\nEvaluating Models")
        train_accuracy = self.model.evaluate(x_train,y_train)
        val_accuracy = self.model.evaluate(x_val,y_val)
        test_accuracy = self.model.evaluate(x_test,y_test)
        print("Accuracy: Train->%.2f | Val->%.2f | Test->%.2f\n"%(train_accuracy, val_accuracy, test_accuracy))
        print("\n\nPredictions")
        predic1 = self.CheckPredictions(x_train, y_train)
        predic2 = self.CheckPredictions(x_val, y_val)
        predic3 = self.CheckPredictions(x_test, y_test)
        predic_str = "Train: " + str(predic1) + "\tVal: " +  str(predic2)  + "\tTest: " + str(predic3) + "\n"
        print(predic_str)
        return "Accuracy: Train->%.2f | Val->%.2f | Test->%.2f\n"%(train_accuracy, val_accuracy, test_accuracy) + predic_str

    def processX(self, X, count):
        new_X = []
        for _ in range(count):
            new_X.append(X[:,_].reshape(100,205))
        return np.array(new_X)
    
    def GetData(self, data_type):
        x, y, size = self.dataset.MiniBatch(-1, data_type)
        x = self.processX(x, size)
        x = x.reshape((size, 100, 205, 1))
        y = y.reshape((size,))
        return x,y

    def CheckPredictions(self, X, Y):
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

    def save(self, filename):
        self.model.save("model" + filename)

    def predict(self, map_sample):
        pass

if __name__ == "__main__":
    dataset_perm = [[70,20,10],[80,10,10],[85,10,5]]
    batch_size = [5,10,20,50]
    #dataset_perm = [[70,20,10]]
    #batch_size = [20]
    for dp in dataset_perm:
        for batchsize in batch_size:
            f = open('performance ' + str(time.asctime()) + '.dat',"a")
            filename = str(time.asctime()) + "_best_model_" + "_".join([str(t) for t in dp]) + "_Batch_" + str(batchsize) + ".hdf5"
            nn = NeuralNet(dp)
            nn.build_model(20500, 1)
            ret = nn.train_model(200,batch_size=batchsize, model_filename=filename)
            f.write(filename + "\t" + ret + "\n")
            del nn
            f.close()