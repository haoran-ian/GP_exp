import numpy as np

class AdvancedPhaseTransitionExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.phase_detection_threshold = 1e-3
        self.exploration_factor = 0.9
        self.exploitation_factor = 0.2
        self.dynamic_ratio = 0.1
        self.elite_fraction = 0.2  # Fraction of top solutions to preserve
        self.adaptive_radius_factor = 0.5  # Adapt radius based on convergence

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

    def explore(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim)) + \
               np.random.normal(0, 0.15, (self.population_size, self.dim))

    def dynamically_adjust_population(self):
        self.population_size = max(int(self.initial_population_size * (1 + self.dynamic_ratio * (self.evaluations / self.budget))), 1)

    def refine_local_search_space(self, best_solution, bounds, radius):
        lower_bound = np.maximum(bounds.lb, best_solution - radius * (bounds.ub - bounds.lb))
        upper_bound = np.minimum(bounds.ub, best_solution + radius * (bounds.ub - bounds.lb))
        return np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))

    def adaptive_search_radius(self, fitness):
        diversity = np.std(fitness)
        return self.adaptive_radius_factor * diversity

    def elite_preservation(self, population, fitness):
        elite_count = max(1, int(self.elite_fraction * len(population)))
        elite_indices = np.argsort(fitness)[:elite_count]
        return population[elite_indices], fitness[elite_indices]

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)

        while self.evaluations < self.budget:
            self.dynamically_adjust_population()
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            if np.random.rand() < 0.6 and self.detect_phase_transition(population):
                new_population = self.explore(bounds)
            else:
                adaptive_radius = self.adaptive_search_radius(fitness)
                new_population = self.refine_local_search_space(best_solution, bounds, adaptive_radius)

            new_fitness = self.evaluate_population(new_population, func)

            # Combine and select the best, preserving elites
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            population, fitness = self.elite_preservation(combined_population, combined_fitness)

        return population[np.argmin(fitness)]