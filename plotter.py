



import matplotlib.pyplot as plt
import numpy as np


def visualize_costmap(pred_base):
    plt.figure(figsize=(6,6))
    plt.imshow(pred_base, cmap='inferno', origin='lower')
    plt.colorbar(label='Cost')
    plt.title("Basic Costmap Visualization")
    plt.show()
