import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def mapOptimalSolution(x,y):

    # Define the function f(x, y)
    def f(x, y):
        return x**2 + y**2 + 1

    # Create a grid of x and y values
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    # Calculate the corresponding z values using the function
    Z = f(X, Y)

   
    ax2.plot_surface(X, Y, Z, cmap='viridis')

    # Label the axes
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('f(X, Y)')

    # Mark a point at (x_point, y_point, z_point)
    x_point = 0.00036
    y_point = 0.0005599
    z_point = f(x_point, y_point)
    ax2.scatter([x_point], [y_point], [z_point], color='blue', s=100, label='Point (2, 3, f(2, 3))')
    # Set the title
    plt.title('Surface Plot of f(X, Y) = X^2 + Y^2 + 1')

    # Show the plot
    plt.show()
