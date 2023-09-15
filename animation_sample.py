import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and axes
fig, ax = plt.subplots()

# Initialize function (clear the plot)
def init():
    ax.clear()

# Update function (plot a sine wave)
def update(frame):
    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(x + 0.1 * frame)
    ax.plot(x, y, label='Sine Wave')
    ax.set_title(f'Frame: {frame}')
    ax.legend()

# Create the animation
ani = FuncAnimation(fig, update, init_func=init, frames=100, interval=100)

# Display the animation or save it as a GIF
plt.show()
# ani.save('sine_wave_animation.gif', writer='pillow')