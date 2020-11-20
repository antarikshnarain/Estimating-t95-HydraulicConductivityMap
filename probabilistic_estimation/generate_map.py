import numpy as np
import csv
from PIL import Image
from matplotlib import cm
import seaborn as sns
import matplotlib.pylab as plt

if __name__ == "__main__":
    filename = "dump.csv"
    f = open(filename)
    reader = csv.reader(f)
    data=[r for r in reader]
    arr = np.array(data, dtype=np.float)
    new_arr = (arr - np.min(arr))/(np.max(arr) - np.min(arr))
    # Lateral inversion along x-axis
    new_arr = np.flip(new_arr, axis=0)
    print(np.min(new_arr))
    print(np.max(new_arr))
    im = Image.fromarray(np.uint8(cm.gist_earth(new_arr)*255))
    im.save("image.png")
    sns.color_palette("rocket_r", as_cmap=True)
    ax = sns.heatmap(new_arr, center=0.5, cmap="icefire_r")
    #plt.show()
    #plt.savefig("map.png")
    #ax.plot()
    #ax.savefig("pi.png")
    f.close()

    f = open("path.csv")
    reader = csv.reader(f)
    matrix = np.zeros((100,206))
    for r in reader:
        for rec in r:
            print(rec)
            x,y = rec.split(":")
            x = int(x)
            y = int(y)
            matrix[x][y] = 1
    matrix = np.flip(matrix, axis=0)
    ax2 = sns.heatmap(matrix, cmap="Oranges")
    f.close()

    plt.show()
