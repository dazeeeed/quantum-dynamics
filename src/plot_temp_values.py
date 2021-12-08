import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt
import pandas as pd
import os

from main import delta_tau, N_STEP

plot_every_x_points = 5

def main():
    x_norm_epsilon = pd.read_csv(os.path.join("..", "data", "temp_values.txt"), sep=';', header=None)
    t = np.array([i*delta_tau*N_STEP for i in range(0, x_norm_epsilon.shape[0])])
    print("Energy max: {:.8e}".format(max(x_norm_epsilon[2][::plot_every_x_points])))

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.set_size_inches(8, 6, forward=True)

    ax1.plot(t[::plot_every_x_points], x_norm_epsilon[0][::plot_every_x_points])
    ax1.set_title("Norm")
    ax2.plot(t[::int(plot_every_x_points/5)], x_norm_epsilon[1][::int(plot_every_x_points/5)])
    ax2.set_title("Mean position")
    ax3.plot(t[::plot_every_x_points], x_norm_epsilon[2][::plot_every_x_points])
    ax3.set_title("Mean energy")

    ax3.set_xlabel("tau")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    main()