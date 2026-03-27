import numpy as np

class AdaptivePhaseTransitionExplorerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.phase_detection_threshold = 1e-3
        self.exploration_factor_start = 0.9
        self.exploration_factor_end = 0.5
        self.exploitation_factor = 0.3
        self.dynamic_ratio = 0.1
        self.variance_threshold = 0.01

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.mean(distances) > self.phase_detection_threshold

    def exploit(self, best_solution, bounds):
        noise = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        return np.clip(best_solution + self.exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds, exploration_factor):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim)) + \
               np.random.normal(0, exploration_factor, (self.population_size, self.dim))

    def dynamically_adjust_population(self, population, fitness):
        if np.var(fitness) < self.variance_threshold:
            self.population_size = max(int(self.population_size * (1 - self.dynamic_ratio)), 1)
        else:
            self.population_size = min(int(self.initial_population_size * (1 + self.dynamic_ratio)), self.initial_population_size)

    def refine_local_search_space(self, best_solution, bounds, radius=0.1):
        lower_bound = np.maximum(bounds.lb, best_solution - radius * (bounds.ub - bounds.lb))
        upper_bound = np.minimum(bounds.ub, best_solution + radius * (bounds.ub - bounds.lb))
        return np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)

        while self.evaluations < self.budget:
            self.dynamically_adjust_population(population, fitness)
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            exploration_factor = self.exploration_factor_start - \
                                 (self.exploration_factor_start - self.exploration_factor_end) * \
                                 (self.evaluations / self.budget)

            if np.random.rand() < 0.6 and self.detect_phase_transition(population):
                new_population = self.explore(bounds, exploration_factor)
            else:
                new_population = self.refine_local_search_space(best_solution, bounds)

            new_fitness = self.evaluate_population(new_population, func)

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[np.argmin(fitness)]