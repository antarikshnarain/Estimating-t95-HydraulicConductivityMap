"""
Author: Antariksh Narain
Description : Train model and visualize performance 
"""

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from NeuralNet import NeuralNet
from datetime import datetime

dropout = 0.5

layers =[
        Conv2D(16, kernel_size=7, padding='same', strides=(2,2), activation='relu', input_shape=(100,205,1)),
        MaxPooling2D(strides=2),
        Conv2D(64, kernel_size=7, padding='same', activation='relu'),
        MaxPooling2D(strides=2),
        Conv2D(256, kernel_size=5, padding='same', activation='relu'),
        MaxPooling2D(strides=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(dropout),
        Dense(256, activation='relu'),
        Dropout(dropout),
        Dense(512, activation='relu'),
        Dense(1, activation='softmax')
]

if __name__ == "__main__":
    #dataset_perm = [[60,20,20],[70,20,10],[80,10,10],[85,10,5]]
    #batch_size = [5,20,50]
    batch_size = [50]
    dataset_perm = [[60,20,20]]
    #loss_types = ['mean_squared_error', 'mean_absolute_percentage_error']
    loss_types = ['mean_squared_error']
    #loss_types = ['mean_absolute_percentage_error']
    #dropouts = [0.25, 0.5]
    dropouts = [0.5]
    for dp in dataset_perm:
        for batchsize in batch_size:
            for loss_type in loss_types:
                for dropout in dropouts:
                    filename = str(datetime.now()) + "_Batch_" + str(batchsize) + "_Loss_" + str(loss_type) + "_D_" + str(dropout)
                    nn = NeuralNet(dp, filename)
                    nn.BuildModel(layers, loss_type=loss_types, dropout=dropout)
                    results = nn.TrainModel(epochs=300,batch_size=batchsize)
                    nn.visualize([1197,2017],save=True)
                    print("=====Performance=====")
                    print("Batch ", batch_size, " Loss ", loss_type, " Dropout ", dropout)
                    print(results)
                    print("---------------------")
                    nn.save()
                    del nn
