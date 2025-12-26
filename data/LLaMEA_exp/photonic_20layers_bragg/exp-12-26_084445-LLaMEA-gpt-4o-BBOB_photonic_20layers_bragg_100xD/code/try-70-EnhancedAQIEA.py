import numpy as np

class EnhancedAQIEA:
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
        self.variance_reduction_factor = 0.95
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5

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

    def adaptive_quantum_update(self, fitness_values):
        normalized_fitness = (fitness_values - fitness_values.min()) / (fitness_values.max() - fitness_values.min() + 1e-6)
        for i in range(self.population_size):
            adjustment = self.adaptive_factor * (1 - normalized_fitness[i])
            variance_adjustment = self.variance_reduction_factor
            if np.random.rand() < self.crossover_rate:
                partner_index = np.random.randint(0, self.population_size)
                self.quantum_states[i] = 0.5 * (self.quantum_states[i] + self.quantum_states[partner_index])
            if np.random.rand() < self.mutation_rate:
                self.quantum_states[i] += variance_adjustment * np.random.normal(0, 1, self.dim)
            self.quantum_states[i] = np.abs(np.clip(self.quantum_states[i], 0, 1))

    def adjust_population_size(self, evaluations):
        progress = evaluations / self.budget
        self.population_size = int(self.min_population_size + (self.max_population_size - self.min_population_size) * (1 - progress))
        self.quantum_states = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.adaptive_factor *= 0.99
        self.variance_reduction_factor *= 0.98

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            self.adjust_population_size(evaluations)
            fitness_values = self.evaluate_population(func)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            self.adaptive_quantum_update(fitness_values)
        return self.best_solution