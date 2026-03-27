import numpy as np

class EnhancedPhaseAdaptiveMultimodalExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.phase_detection_threshold = 0.01  # Adjusted for better phase transition detection
        self.exploration_factor = 1.0  # Increased to enhance searching capability
        self.exploitation_factor = 0.2  # Slightly decreased to balance exploration
        self.dynamic_ratio = 0.2  # Increased to allow more population adjustment
        self.local_search_radius = 0.05  # Reduced for more precise local search

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.std(distances) > self.phase_detection_threshold  # Using standard deviation for more robust detection

    def exploit(self, best_solution, bounds):
        noise = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        return np.clip(best_solution + self.exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim)) + \
               np.random.normal(0, 0.05, (self.population_size, self.dim))  # Further reduced normal distribution std

    def dynamically_adjust_population(self):
        self.population_size = max(int(self.initial_population_size * (1 + self.dynamic_ratio * (1 - (self.evaluations / self.budget)))), 1)

    def refine_local_search_space(self, best_solution, bounds):
        lower_bound = np.maximum(bounds.lb, best_solution - self.local_search_radius * (bounds.ub - bounds.lb))
        upper_bound = np.minimum(bounds.ub, best_solution + self.local_search_radius * (bounds.ub - bounds.lb))
        return np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)

        while self.evaluations < self.budget:
            self.dynamically_adjust_population()
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            if np.random.rand() < 0.5 and self.detect_phase_transition(population):  # Adjusted probability for switching
                new_population = self.explore(bounds)
            else:
                new_population = self.refine_local_search_space(best_solution, bounds)

            new_fitness = self.evaluate_population(new_population, func)

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[np.argmin(fitness)]