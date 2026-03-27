import numpy as np

class EnhancedAdaptivePhaseTransitionExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 50
        self.phase_detection_threshold = 1e-3
        self.exploration_factor = 0.9
        self.exploitation_factor = 0.2
        self.memory_factor = 0.1  # Factor for adaptive memory influence
        self.multi_scale_factor = [0.5, 1.0, 1.5]  # Multi-scale exploration factors

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
        noise = np.random.uniform(-1, 1, (self.population_size, self.dim))
        return np.clip(best_solution + self.exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds, scale_factor):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim)) + \
               np.random.normal(0, 0.1 * scale_factor, (self.population_size, self.dim))

    def adaptive_memory(self, prev_population, current_population):
        return (1 - self.memory_factor) * current_population + self.memory_factor * prev_population

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)
        prev_population = population.copy()

        while self.evaluations < self.budget:
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            if np.random.rand() < 0.5 and self.detect_phase_transition(population):  # Probabilistic phase switching
                scale_factor = np.random.choice(self.multi_scale_factor)
                new_population = self.explore(bounds, scale_factor)
            else:
                new_population = self.exploit(best_solution, bounds)

            new_population = self.adaptive_memory(prev_population, new_population)
            new_fitness = self.evaluate_population(new_population, func)

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            prev_population = population
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[np.argmin(fitness)]