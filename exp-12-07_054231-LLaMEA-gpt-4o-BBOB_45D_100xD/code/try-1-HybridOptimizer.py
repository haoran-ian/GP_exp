import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.mutation_factor = 0.8
        self.crossover_probability = 0.7
        self.temperature = 1.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Differential Evolution mutation
                self.mutation_factor = 0.5 + np.random.rand() * 0.5  # Adaptive mutation factor
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                self.crossover_probability = 0.6 + np.random.rand() * 0.4  # Adaptive crossover probability
                cross_points = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Simulated Annealing acceptance criterion
                    if trial_fitness < best_fitness or np.exp((best_fitness - trial_fitness) / self.temperature) > np.random.rand():
                        best_solution = trial
                        best_fitness = trial_fitness

                # Cooling
                self.temperature *= self.cooling_rate

        return best_solution