import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt
import pandas as pd
import os

from main import delta_tau

def main():
    x_norm_epsilon = pd.read_csv(os.path.join("..", "data", "temp_values.txt"), sep=';', header=None)
    t = np.array([i*delta_tau for i in range(0, x_norm_epsilon.shape[0])])


    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.set_size_inches(8, 6, forward=True)

    ax1.plot(t, x_norm_epsilon[0])
    ax1.set_title("Norm")
    ax2.plot(t, x_norm_epsilon[1])
    ax2.set_title("Mean position")
    ax3.plot(t, x_norm_epsilon[2])
    ax3.set_title("Mean energy")

    ax3.set_xlabel("tau")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    main()