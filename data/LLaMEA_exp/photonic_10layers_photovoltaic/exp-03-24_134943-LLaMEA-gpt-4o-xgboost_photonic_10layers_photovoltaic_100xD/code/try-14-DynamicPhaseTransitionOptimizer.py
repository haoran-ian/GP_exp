import numpy as np

class DynamicPhaseTransitionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.phase_detection_threshold = 1e-3
        self.exploration_factor = 0.9
        self.exploitation_factor = 0.2
        self.importance_factor = 0.5

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.mean(distances) > self.phase_detection_threshold

    def adjust_population_size(self, phase_detected):
        if phase_detected:
            self.population_size = int(self.initial_population_size * 1.5)
        else:
            self.population_size = self.initial_population_size

    def exploit(self, best_solution, bounds):
        noise = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        return np.clip(best_solution + self.exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim)) + \
               np.random.normal(0, 0.15, (self.population_size, self.dim))

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)

        while self.evaluations < self.budget:
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            phase_detected = self.detect_phase_transition(population)
            self.adjust_population_size(phase_detected)

            if np.random.rand() < 0.6 * self.importance_factor:
                new_population = self.explore(bounds)
            else:
                new_population = self.exploit(best_solution, bounds)

            new_fitness = self.evaluate_population(new_population, func)

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[np.argmin(fitness)]