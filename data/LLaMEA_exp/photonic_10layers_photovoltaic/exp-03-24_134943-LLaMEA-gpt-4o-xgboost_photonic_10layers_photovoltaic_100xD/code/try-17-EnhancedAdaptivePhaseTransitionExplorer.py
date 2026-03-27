import numpy as np

class EnhancedAdaptivePhaseTransitionExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = 50
        self.current_population_size = self.initial_population_size
        self.phase_detection_threshold = 1e-3
        self.exploration_factor = 0.9
        self.exploitation_factor = 0.2
        self.learning_rate = 0.1

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.current_population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.mean(distances) > self.phase_detection_threshold

    def exploit(self, best_solution, bounds):
        noise = np.random.uniform(-0.5, 0.5, (self.current_population_size, self.dim))
        return np.clip(best_solution + self.exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.current_population_size, self.dim)) + \
               np.random.normal(0, 0.15, (self.current_population_size, self.dim))

    def adjust_learning_rate(self, previous_fitness, current_fitness):
        improvement = np.min(previous_fitness) - np.min(current_fitness)
        self.learning_rate = max(0.01, min(0.5, self.learning_rate + 0.1 * improvement))
        self.current_population_size = max(10, int(self.initial_population_size * (1 + self.learning_rate)))

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)

        while self.evaluations < self.budget:
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            if np.random.rand() < 0.6 and self.detect_phase_transition(population):
                new_population = self.explore(bounds)
            else:
                new_population = self.exploit(best_solution, bounds)

            new_fitness = self.evaluate_population(new_population, func)

            self.adjust_learning_rate(fitness, new_fitness)

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.current_population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[np.argmin(fitness)]