import numpy as np

class DynamicMultimodalPhaseNavigator:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 50
        self.phase_detection_threshold = 1e-3
        self.exploration_factor = 0.8
        self.exploitation_factor = 0.2
        self.diversity_threshold = 0.1

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.mean(distances) > self.phase_detection_threshold

    def measure_diversity(self, population):
        centroid = np.mean(population, axis=0)
        return np.mean(np.linalg.norm(population - centroid, axis=1))

    def adapt_exploration_exploitation(self, population, fitness):
        if self.measure_diversity(population) < self.diversity_threshold:
            self.exploration_factor *= 1.1
            self.exploitation_factor *= 0.9
        else:
            self.exploration_factor *= 0.9
            self.exploitation_factor *= 1.1
        self.exploration_factor = min(self.exploration_factor, 1.0)
        self.exploitation_factor = min(self.exploitation_factor, 1.0)

    def exploit(self, best_solution, bounds):
        noise = np.random.uniform(-1, 1, (self.population_size, self.dim))
        return np.clip(best_solution + self.exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)

        while self.evaluations < self.budget:
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            if self.detect_phase_transition(population):
                # Explore phase
                new_population = self.explore(bounds)
            else:
                # Exploit phase
                new_population = self.exploit(best_solution, bounds)

            new_fitness = self.evaluate_population(new_population, func)

            # Selection
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            # Adapt exploration and exploitation factors
            self.adapt_exploration_exploitation(population, fitness)

        return population[np.argmin(fitness)]