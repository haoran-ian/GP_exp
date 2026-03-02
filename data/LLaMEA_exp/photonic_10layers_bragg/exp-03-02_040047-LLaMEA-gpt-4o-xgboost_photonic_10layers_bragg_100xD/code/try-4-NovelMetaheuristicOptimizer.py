import numpy as np

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50  # Starting population size
        self.elite_fraction = 0.1  # Fraction of elite solutions
        self.exploration_factor = 0.35  # Exploration threshold
        self.exploitation_factor = 0.65  # Exploitation threshold

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        
        while evaluations < self.budget:
            elite_count = int(self.population_size * self.elite_fraction)
            elite_solutions = population[np.argsort(fitness)[:elite_count]]

            if np.random.rand() < self.exploration_factor:
                new_population = np.random.uniform(lb, ub, (self.population_size, self.dim))
            else:
                perturbation = np.random.normal(loc=0.0, scale=0.1, size=(self.population_size, self.dim))
                new_population = elite_solutions[np.random.randint(elite_count, size=self.population_size)] + perturbation
                new_population = np.clip(new_population, lb, ub)
            
            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += self.population_size

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))

            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_solution = population[np.argmin(fitness)]
            
        return best_solution