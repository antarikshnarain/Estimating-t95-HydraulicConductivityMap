"""
Author: Antariksh Narain
Description: Utility function for Neural Network with Keras
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from keras import models
import os

class Utility:
    SUMMARY = "summmary/"
    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(self.SUMMARY):
            os.makedirs(self.SUMMARY)

    def summarize(self, model_summary):
        f = open(self.SUMMARY + self.filename, "w")
        print(model_summary)
        f.write(model_summary)
        f.close()
    
    def display_accuracy(self, epochs, accuracy, val_accuracy):
        plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'g', label='Validation Accuracy')
        plt.title("Train & Val Accuracy " + self.filename)
        plt.legend()
        plt.figure()
        #plt.savefig(self.filename + "_accuracy.png")
    
    def display_loss(self, epochs, loss, val_loss):
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'g', label='Validation Loss')
        plt.title("Train & Val Loss " + self.filename)
        plt.legend()
        plt.figure()
        #plt.savefig(self.filename + "_loss.png")
        plt.show()

    def show_layers(self, classifier, img_map):
        layer_outputs = [layer.output for layer in classifier.layers[:12]] 
        # Extracts the outputs of the top 12 layers
        # Creates a model that will return these outputs, given the model input
        activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
        activations = activation_model.predict(img_map)
        layer_names = []
        for layer in classifier.layers[:12]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
            
        images_per_row = 16
        for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
            n_features = layer_activation.shape[-1] # Number of features in the feature map
            size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
            n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                                    :, :,
                                                    col * images_per_row + row]
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size, # Displays the grid
                                row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            #plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.savefig(layer_name + layer_activation + ".png")