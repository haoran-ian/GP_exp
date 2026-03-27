import numpy as np

class EnhancedAdaptivePhaseTransitionExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 50
        self.phase_detection_threshold = 1e-3
        self.exploration_factor = 0.8
        self.exploitation_factor = 0.2
        self.memory_factor = 0.5
        self.memory_population = []

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

    def explore(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def memory_influenced_search(self, bounds):
        if self.memory_population:
            memory_array = np.array(self.memory_population)
            memory_best = memory_array[np.argmin([func(x) for x in memory_array])]
            memory_noise = np.random.uniform(-1, 1, (self.population_size, self.dim))
            return np.clip(memory_best + self.memory_factor * memory_noise, bounds.lb, bounds.ub)
        else:
            return self.explore(bounds)

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)

        while self.evaluations < self.budget:
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            self.memory_population.append(best_solution)

            if self.detect_phase_transition(population):
                # Explore phase with memory influence
                new_population = self.memory_influenced_search(bounds)
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

        return population[np.argmin(fitness)]