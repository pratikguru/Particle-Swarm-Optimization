import random as rd
import numpy as np
import matplotlib.pyplot as plt

rd.seed(12)

# PSO parameters
W: float = 0.5  # Inertia weight
c1: float = 0.8  # Personal best weight
c2: float = 0.9  # Global best weight

n_iterations: int = 100  # Maximum number of iterations
n_particles: int = 50     # Number of particles in the swarm
target_error: float = 1e-6  # Target error for stopping criteria

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)

ax2 = fig.add_subplot(122, projection='3d')
class Particle:
    """Class representing a particle in the swarm."""

    def __init__(self) -> None:
        """Initialize a particle with random position and velocity."""
        x = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
        y = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
        self.position: np.ndarray[float] = np.array([x, y])
        self.pBest_position: np.ndarray[float] = self.position
        self.pBest_value: float = float('inf')
        self.velocity: np.ndarray[float] = np.array([0, 0])

    def update(self) -> None:
        """Update the particle's position based on its velocity."""
        self.position = self.position + self.velocity

class Space:
    """Class managing the optimization process."""

    def __init__(self, target: float, target_error: float, n_particles: int) -> None:
        """Initialize the search space and particles.

        Args:
            target (float): The target value to reach.
            target_error (float): Target error for stopping criteria.
            n_particles (int): Number of particles in the swarm.
        """
        self.target: float = target
        self.target_error: float = target_error
        self.n_particles: int = n_particles
        self.particles: list[Particle] = []
        self.gBest_value: float = float('inf')
        self.gBest_position: np.ndarray[float] = np.array([rd.random() * 50, rd.random() * 50])

    def fitness(self, particle: Particle) -> float:
        """Evaluate the fitness of a particle's position.

        Args:
            particle (Particle): The particle to evaluate.

        Returns:
            float: The fitness value.
        """
        x, y = particle.position
        return x**2 + y**2 + 1

    def set_pBest(self) -> None:
        """Update each particle's personal best (pBest)."""
        for particle in self.particles:
            fitness_candidate: float = self.fitness(particle)
            if particle.pBest_value > fitness_candidate:
                particle.pBest_value = fitness_candidate
                particle.pBest_position = particle.position

    def set_gBest(self) -> None:
        """Update the global best (gBest) among all particles."""
        for particle in self.particles:
            best_fitness_candidate: float = self.fitness(particle)
            if self.gBest_value > best_fitness_candidate:
                self.gBest_value = best_fitness_candidate
                self.gBest_position = particle.position

    def update_particles(self) -> None:
        """Update particle velocities and positions."""
        global W
        for particle in self.particles:
            inertial: np.ndarray[float] = W * particle.velocity
            self_confidence: np.ndarray[float] = c1 * rd.random() * (particle.pBest_position - particle.position)
            swarm_confidence: np.ndarray[float] = c2 * rd.random() * (self.gBest_position - particle.position)
            new_velocity: np.ndarray[float] = inertial + self_confidence + swarm_confidence
            particle.velocity = new_velocity
            particle.update()

    def show_particles(self, iteration: int) -> None:
        """Display particle positions for visualization.

        Args:
            iteration (int): Current iteration count.
        """
        print(iteration, 'iterations')
        print('BestPosition in this time:', self.gBest_position)
        print('BestValue in this time:', self.gBest_value)

        plt.title("Best Position")
        plt.ylim(-800, 800)
        plt.xlim(-800, 800)
        
  
        for particle in self.particles:

            ax.plot(self.gBest_position[0], self.gBest_position[1], 'bo')
            ax.plot(particle.position[0], particle.position[1], 'ro')
            
        plt.pause(0.00001)
        plt.show(block=False)

def create_particle() -> Particle:
    """Create a new particle with random initial position."""
    x = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
    y = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
    return Particle()

def mapOptimalSolution(px,py):

    # Define the function f(x, y)
    def f(x, y):
        return x**2 + y**2 + 1

    # Create a grid of x and y values
    x = np.linspace(-500, 500, 100)
    y = np.linspace(-500, 500, 100)
    X, Y = np.meshgrid(x, y)
    # Calculate the corresponding z values using the function
    Z = f(X, Y)

    ax2.clear()
    ax2.plot_surface(X, Y, Z, cmap='viridis')

    # Label the axes
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('f(X, Y)')
    plt.xlim(-800, 800)
    plt.ylim(-800, 800)

    # Mark a point at (x_point, y_point, z_point)
    x_point = px
    y_point = py
    z_point = f(x_point, y_point)
    ax2.plot_surface(X, Y, Z, cmap='viridis')
    ax2.scatter([x_point], [y_point], [0], color='blue', s=100, label='Point (2, 3, f(2, 3))')
    # Set the title
    plt.title('Surface Plot of f(X, Y) = X^2 + Y^2 + 1')

    # Show the plot
    plt.show(block=False)
    plt.pause(0.00000000001)


def main() -> None:
    """Main function to perform particle swarm optimization."""
    search_space: Space = Space(1, target_error, n_particles)
    search_space.particles = [create_particle() for _ in range(search_space.n_particles)]

    iteration: int = 0
    while iteration < n_iterations:
        search_space.set_pBest()
        search_space.set_gBest()

        ax.clear()
       
        search_space.show_particles(iteration)

        if abs(search_space.gBest_value - search_space.target) <= search_space.target_error:
            break
        
        search_space.update_particles()
        mapOptimalSolution(search_space.gBest_position[0], search_space.gBest_position[1])
        iteration += 1
    
    plt.pause(1000000000)
    plt.tight_layout()
    print("The best solution is:", search_space.gBest_position, "in", iteration, "iterations")

if __name__ == "__main__":
    main()
