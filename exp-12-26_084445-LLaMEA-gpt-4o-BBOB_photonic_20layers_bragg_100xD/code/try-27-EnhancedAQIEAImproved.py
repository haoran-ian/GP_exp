import numpy as np

class EnhancedAQIEAImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.min_population_size = 10
        self.max_population_size = 100
        self.population_size = self.initial_population_size
        self.quantum_states = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_factor = 0.05
        self.elite_ratio = 0.1  # Proportion of elite individuals to preserve

    def collapse_state(self, state, bounds):
        return bounds.lb + (bounds.ub - bounds.lb) * state

    def evaluate_population(self, func):
        fitness_values = np.zeros(self.population_size)
        for i in range(self.population_size):
            solution = self.collapse_state(self.quantum_states[i], func.bounds)
            fitness_values[i] = func(solution)
            if fitness_values[i] < self.best_fitness:
                self.best_fitness = fitness_values[i]
                self.best_solution = solution
        return fitness_values

    def preserve_elite(self, fitness_values):
        elite_count = max(1, int(self.elite_ratio * self.population_size))
        elite_indices = np.argsort(fitness_values)[:elite_count]
        return self.quantum_states[elite_indices]

    def adaptive_quantum_update(self, fitness_values, elites):
        variance = np.var(fitness_values)
        rotation_factor = self.adaptive_factor * (1 / (1 + variance))
        normalized_fitness = (fitness_values - fitness_values.min()) / (fitness_values.max() - fitness_values.min() + 1e-6)

        for i in range(self.population_size):
            if i not in elites:
                adjustment = rotation_factor * (1 - normalized_fitness[i])
                if np.random.rand() < 0.5:
                    self.quantum_states[i] = np.abs(np.sin(np.pi * (normalized_fitness[i] + adjustment) * self.quantum_states[i] +
                                                            (1 - (normalized_fitness[i] + adjustment)) * np.random.rand(self.dim)))
                else:
                    self.quantum_states[i] = np.abs(np.cos(np.pi * (normalized_fitness[i] + adjustment) * self.quantum_states[i] +
                                                            (1 - (normalized_fitness[i] + adjustment)) * np.random.rand(self.dim)))

    def adjust_population_size(self, evaluations):
        progress = evaluations / self.budget
        self.population_size = int(self.min_population_size + (self.max_population_size - self.min_population_size) * (1 - progress))
        self.quantum_states = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.adaptive_factor *= 0.99

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            self.adjust_population_size(evaluations)
            fitness_values = self.evaluate_population(func)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            elites = self.preserve_elite(fitness_values)
            self.adaptive_quantum_update(fitness_values, elites)
        return self.best_solution