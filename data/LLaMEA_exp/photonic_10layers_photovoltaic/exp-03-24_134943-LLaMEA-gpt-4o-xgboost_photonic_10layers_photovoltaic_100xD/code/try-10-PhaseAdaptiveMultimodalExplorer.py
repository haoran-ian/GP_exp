import numpy as np

class PhaseAdaptiveMultimodalExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.base_population_size = 50
        self.phase_detection_threshold = 1e-3
        self.exploration_factor = 0.8
        self.exploitation_factor = 0.2

    def initialize_population(self, bounds, population_size):
        return np.random.uniform(bounds.lb, bounds.ub, (population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.mean(distances) > self.phase_detection_threshold

    def exploit(self, best_solution, bounds, population_size):
        noise_scale = np.random.exponential(scale=1.0, size=(population_size, self.dim))
        noise_direction = np.random.normal(0, 1, (population_size, self.dim))
        noise = noise_scale * noise_direction
        return np.clip(best_solution + self.exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds, population_size):
        return np.random.uniform(bounds.lb, bounds.ub, (population_size, self.dim))

    def adapt_population_size(self, fitness, base_size):
        diversity = np.std(fitness)
        return int(base_size * (1 + diversity))

    def __call__(self, func):
        bounds = func.bounds
        population_size = self.base_population_size
        population = self.initialize_population(bounds, population_size)
        fitness = self.evaluate_population(population, func)

        while self.evaluations < self.budget:
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            if self.detect_phase_transition(population):
                # Explore phase
                new_population = self.explore(bounds, population_size)
            else:
                # Exploit phase
                new_population = self.exploit(best_solution, bounds, population_size)

            new_fitness = self.evaluate_population(new_population, func)

            # Selection
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            # Adaptive population size
            population_size = self.adapt_population_size(fitness, self.base_population_size)

        return population[np.argmin(fitness)]