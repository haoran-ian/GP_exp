import numpy as np

class DynamicMultimodalExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = 50
        self.dynamic_population_size = self.initial_population_size
        self.phase_detection_threshold = 1e-2
        self.exploration_factor = 0.85
        self.exploitation_factor = 0.3
        self.dynamic_ratio = 0.15
        self.exploration_decay = 0.995
        self.last_best_fitness = float('inf')
        self.stagnation_counter = 0
        self.stagnation_threshold = 10

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.dynamic_population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.std(distances) > self.phase_detection_threshold

    def exploit(self, best_solution, bounds):
        noise = np.random.normal(0, 1, (self.dynamic_population_size, self.dim))
        return np.clip(best_solution + self.exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.dynamic_population_size, self.dim)) * \
               np.random.normal(1, self.exploration_decay, (self.dynamic_population_size, self.dim))

    def dynamically_adjust_population(self):
        self.dynamic_population_size = max(int(self.initial_population_size * (1 + self.dynamic_ratio * (self.evaluations / self.budget))), 1)

    def refine_local_search_space(self, best_solution, bounds, radius=0.1):
        lower_bound = np.maximum(bounds.lb, best_solution - radius * (bounds.ub - bounds.lb))
        upper_bound = np.minimum(bounds.ub, best_solution + radius * (bounds.ub - bounds.lb))
        return np.random.uniform(lower_bound, upper_bound, (self.dynamic_population_size, self.dim))

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)

        while self.evaluations < self.budget:
            self.dynamically_adjust_population()
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            best_fitness = fitness[best_idx]

            if best_fitness >= self.last_best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            self.last_best_fitness = best_fitness

            if np.random.rand() < 0.6 and (self.detect_phase_transition(population) or self.stagnation_counter > self.stagnation_threshold):
                new_population = self.explore(bounds)
            else:
                new_population = self.refine_local_search_space(best_solution, bounds)

            new_fitness = self.evaluate_population(new_population, func)

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.dynamic_population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[np.argmin(fitness)]