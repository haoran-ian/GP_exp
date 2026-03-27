import numpy as np

class DynamicMultimodalPhaseTransitionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.phase_detection_threshold = 1e-3
        self.exploration_factor = 1.0  # Increased exploration for diverse search
        self.exploitation_factor = 0.3  # Fine-tuned exploitation for precision
        self.dynamic_ratio = 0.15  # More aggressive dynamic population adjustment
        self.diversity_maintain_factor = 0.1  # Factor to maintain diversity

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
               np.random.normal(0, 0.05, (self.population_size, self.dim))  # Further reduced normal distribution std deviation

    def dynamically_adjust_population(self):
        self.population_size = max(int(self.initial_population_size * (1 + self.dynamic_ratio * (self.evaluations / self.budget))), 1)

    def refine_local_search_space(self, best_solution, bounds, radius=0.1):
        lower_bound = np.maximum(bounds.lb, best_solution - radius * (bounds.ub - bounds.lb))
        upper_bound = np.minimum(bounds.ub, best_solution + radius * (bounds.ub - bounds.lb))
        return np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))

    def maintain_diversity(self, population, bounds):
        diversity_noise = np.random.normal(0, self.diversity_maintain_factor, (self.population_size, self.dim))
        new_population = population + diversity_noise
        return np.clip(new_population, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)

        while self.evaluations < self.budget:
            self.dynamically_adjust_population()
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            if np.random.rand() < 0.7 and self.detect_phase_transition(population):
                new_population = self.explore(bounds)
            else:
                new_population = self.refine_local_search_space(best_solution, bounds)

            new_population = self.maintain_diversity(new_population, bounds)
            new_fitness = self.evaluate_population(new_population, func)

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[np.argmin(fitness)]