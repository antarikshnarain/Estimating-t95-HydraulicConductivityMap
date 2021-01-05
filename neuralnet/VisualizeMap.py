import numpy as np
import csv
from PIL import Image
from matplotlib import cm
import seaborn as sns
import matplotlib.pylab as plt

from keract import get_activations, display_activations

class Visualize:
    def __init__(self):
        #sns.color_palette("rocket_r", as_cmap=True)
        pass

    def plot(self, map_array):
        map_array = map_array.reshape((100,205))
        map_array = (map_array - np.min(map_array))/ (np.max(map_array) - np.min(map_array))
        ax = sns.heatmap(map_array, center=0.5, cmap="icefire_r")
    
    def save(self, filename="map.png"):    
        plt.savefig(filename)

class VisualizeLayers:
    def __init__(self):
        pass

    def visualize(self, model,layer_input,save_image=False, path='.'):
        #keract_inputs = layer_input[:1]
        keract_inputs = layer_input
        activations = get_activations(model, keract_inputs)
        display_activations(activations, save=save_image, directory=path,cmap='icefire_r')
        