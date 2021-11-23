import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def main():
    rho = pd.read_csv(os.path.join("..", "data", "rho.txt"), sep=';', header=None)
    
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(0,5))
    line, = ax.plot([], [], lw=3)


    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = np.linspace(0, 1, 101)
        y = rho.iloc[i, :]
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init,
                                frames=rho.shape[0], interval=2, blit=True)


    save_animation = False
    if save_animation:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist='01141448'), bitrate=-1, extra_args=['-vcodec', 'libx264'])
        anim.save('wave_function.mp4', writer=writer)

    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    main()