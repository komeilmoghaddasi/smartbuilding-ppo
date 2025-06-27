# utils.py

import matplotlib.pyplot as plt

def plot_metric(metric_name, x_data, y_data, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, marker='o', linestyle='-', linewidth=1.5, label=metric_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
