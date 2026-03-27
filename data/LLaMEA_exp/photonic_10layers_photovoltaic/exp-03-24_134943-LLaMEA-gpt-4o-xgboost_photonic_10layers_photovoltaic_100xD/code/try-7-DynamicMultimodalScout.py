import numpy as np

class DynamicMultimodalScout:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 50
        self.phase_detection_threshold = 1e-3
        self.exploration_factor = 0.8
        self.exploitation_factor = 0.2
        self.learning_rate = 0.1  # Adaptive learning rate for phase transition detection
        self.diversity_threshold = 0.1  # Threshold to maintain population diversity

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        # Calculate diversity to update the phase detection threshold adaptively
        diversity = np.std(population, axis=0).mean()
        self.phase_detection_threshold = max(self.phase_detection_threshold, diversity * self.learning_rate)
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.mean(distances) > self.phase_detection_threshold

    def exploit(self, best_solution, bounds):
        noise = np.random.normal(0, 1, (self.population_size, self.dim))
        return np.clip(best_solution + self.exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def maintain_diversity(self, population, bounds):
        # Add random individuals if diversity is below the threshold
        diversity = np.std(population, axis=0).mean()
        if diversity < self.diversity_threshold:
            num_new_individuals = int(self.population_size * 0.2)
            new_individuals = np.random.uniform(bounds.lb, bounds.ub, (num_new_individuals, self.dim))
            population = np.vstack((population, new_individuals))
        return population

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

            new_population = self.maintain_diversity(new_population, bounds)
            new_fitness = self.evaluate_population(new_population, func)

            # Selection
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[np.argmin(fitness)]