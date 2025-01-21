import numpy as np
from scipy.special import gamma  # Importez gamma depuis scipy.special

class FPA:
    def __init__(self, dimension, bounds, pop_size=50, epoch=1000, p_s=0.8, levy_multiplier=1.5):
        self.dimension = dimension
        self.bounds = bounds
        self.pop_size = pop_size
        self.epoch = epoch
        self.p_s = p_s
        self.levy_multiplier = levy_multiplier
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self):
        return [np.random.uniform(self.bounds[0], self.bounds[1], self.dimension) for _ in range(self.pop_size)] #random ab

    def amend_solution(self, solution):
        return np.clip(solution, self.bounds[0], self.bounds[1])

    def levy_flight(self):
        beta = 1.5
        
        # Utilisez gamma depuis scipy.special
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=self.dimension)
        v = np.random.normal(0, 1, size=self.dimension)
        step = u / abs(v)**(1 / beta)
        return step

    def evaluate_population(self, objective_function):
        fitness = [objective_function(ind) for ind in self.population]
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_solution = self.population[best_idx]

    def evolve(self, objective_function):
        # Évaluez la population initiale pour définir la meilleure solution
        self.evaluate_population(objective_function)

        for _ in range(self.epoch):
            new_population = []
            for i in range(self.pop_size):
                if np.random.rand() < self.p_s:  # Global pollination random01
                    levy_step = self.levy_flight() * self.levy_multiplier
                    new_solution = self.population[i] + levy_step * (self.best_solution - self.population[i])
                else:  # Local pollination
                    idx1, idx2 = np.random.choice(range(self.pop_size), 2, replace=False)
                    new_solution = self.population[i] + np.random.rand() * (self.population[idx1] - self.population[idx2])
                new_solution = self.amend_solution(new_solution)
                new_population.append(new_solution)
            self.population = new_population
            self.evaluate_population(objective_function)


# Functions for optimization

def rosenbrock(solution):
    return sum(100 * (solution[j+1] - solution[j]**2)**2 + (1 - solution[j])**2 for j in range(len(solution)-1))

def rastrigin(solution):
    return 10 * len(solution) + sum(x**2 - 10 * np.cos(2 * np.pi * x) for x in solution)

def ackley(solution):
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(solution)
    sum1 = sum(x**2 for x in solution)
    sum2 = sum(np.cos(c * x) for x in solution)
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

def schwefel(solution):
    return 418.9829 * len(solution) - sum(x * np.sin(np.sqrt(abs(x))) for x in solution)

# Example usage
dimension = 100
bounds = (-10, 10)
pop_size = 30
epoch = 1000
levy_multiplier=1.5
p_s=0.8

# Select the function to optimize
objective_function = rosenbrock  # Change to rastrigin, ackley, or schwefel as needed

fpa = FPA(dimension=dimension, bounds=bounds, pop_size=pop_size, epoch=epoch,p_s=p_s,levy_multiplier=levy_multiplier)
fpa.evolve(objective_function)

print(f"Best solution: {fpa.best_solution}")
print(f"Best fitness: {fpa.best_fitness}")


