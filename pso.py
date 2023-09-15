"""
Build: docker build -t pos .
Run: docker run -v output:/app/output pso 
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime

# Get the current date and time
current_datetime = datetime.datetime.now()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the Sphere function for optimization
def Sphere(x):
    return np.sum(np.square(x))

# Input parameters for the Particle Swarm Optimization (PSO) algorithm
d = 10  # Dimensionality of the problem
xMin, xMax = -100, 100  # Search space boundaries
vMin, vMax = -0.2 * (xMax - xMin), 0.2 * (xMax - xMin)  # Velocity boundaries
MaxIt = 3000  # Maximum number of iterations
ps = 10  # Number of particles
c1 = 2  # Cognitive coefficient
c2 = 2  # Social coefficient
w = 0.9 - ((0.9 - 0.4) / MaxIt) * np.linspace(0, MaxIt, MaxIt)  # Inertia weight

# Particle Swarm Optimization Algorithm
def Optimization():
    # Define the Particle class
    class Particle():
        def __init__(self):
            # Initialize particle positions and velocities randomly within the search space
            self.position = np.random.uniform(xMin, 50, [ps, d])
            self.velocity = np.random.uniform(vMin, vMax, [ps, d])

            # Initialize particle cost and personal best
            self.cost = np.zeros(ps)
            self.cost[:] = Sphere(self.position[:])
            self.pbest = np.copy(self.position)
            self.pbest_cost = np.copy(self.cost)
            self.index = np.argmin(self.pbest_cost)

            # Initialize global best
            self.gbest = self.pbest[self.index]
            self.gbest_cost = self.pbest_cost[self.index]
            self.bestCost = np.zeros(MaxIt)

        # Limit particle velocities within specified bounds
        def limitV(self, V):
            for i in range(len(V)):
                if V[i] > vMax:
                    V[i] = vMax
                if V[i] < vMin:
                    V[i] = vMin
            return V

        # Limit particle positions within specified bounds
        def limitX(self, X):
            for i in range(len(X)):
                if X[i] > xMax:
                    X[i] = xMax
                if X[i] < xMin:
                    X[i] = xMin
            return X

        # Evaluate the particles and perform PSO optimization
        def Evaluate(self):
            logging.info("Starting PSO optimization")
            for it in range(MaxIt):
                for i in range(ps):
                    # Update particle velocity
                    self.velocity[i] = (
                        w[it] * self.velocity[i]
                        + c1 * np.random.rand(d) * (self.pbest[i] - self.position[i])
                        + c2 * np.random.rand(d) * (self.gbest - self.position[i])
                    )
                    self.velocity[i] = self.limitV(self.velocity[i])

                    # Update particle position
                    self.position[i] = self.position[i] + self.velocity[i]
                    self.position[i] = self.limitX(self.position[i])

                    # Evaluate the new position
                    self.cost[i] = Sphere(self.position[i])

                    # Update personal best and global best
                    if self.cost[i] < self.pbest_cost[i]:
                        self.pbest[i] = self.position[i]
                        self.pbest_cost[i] = self.cost[i]

                        if self.pbest_cost[i] < self.gbest_cost:
                            self.gbest = self.pbest[i]
                            self.gbest_cost = self.pbest_cost[i]

                self.bestCost[it] = self.gbest_cost

        # Plot the optimization results
        def Plot(self):
            plt.semilogy(self.bestCost)
            plt.xlabel("Iteration Count (N)")
            plt.ylabel("Best Function Value")
            plt.title("PSO with Sphere Function")
            print("Optimal Fitness Value: ", self.gbest_cost)
            plt.plot(self.gbest_cost, '-o')

            # Save the plot with a timestamp in the filename
            current_time = datetime.datetime.now().time()
            plt.savefig(f"./output/output-{current_time}.png")
            plt.show()

    # Create a Particle object and perform PSO optimization
    p = Particle()
    p.Evaluate()
    p.Plot()

# Entry point of the program
if __name__ == "__main__":
    Optimization()
