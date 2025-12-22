import numpy as np

class EAQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.population_size = self.initial_population_size
        self.q_population = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_rate = 0.05

    def _quantum_observation(self):
        angles = np.arccos(1 - 2 * self.q_population)
        return (np.cos(angles) > np.random.rand(*angles.shape)).astype(float)

    def _evaluate_population(self, func, bounds):
        real_population = bounds.lb + self._quantum_observation() * (bounds.ub - bounds.lb)
        fitness = np.array([func(ind) for ind in real_population])
        return real_population, fitness

    def _update_best(self, real_population, fitness):
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_solution = real_population[min_idx]

    def _adaptive_mutation(self):
        mutation_strength = np.abs(np.random.normal(0, self.mutation_rate, self.q_population.shape))
        adapt_factor = np.random.rand(*self.q_population.shape)
        self.q_population += mutation_strength * (adapt_factor - 0.5)
        self.q_population = np.clip(self.q_population, 0, 1)

    def _dynamic_population_size(self, evaluations):
        progress_rate = evaluations / self.budget
        new_size = int(self.initial_population_size * (1 - progress_rate) + 5)
        if new_size != self.population_size:
            self.population_size = new_size
            self.q_population = np.random.uniform(0, 1, (self.population_size, self.dim))

    def _adjust_mutation_rate(self, evaluations):
        progress_rate = evaluations / self.budget
        self.mutation_rate = 0.05 * (1 - progress_rate) + 0.01

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0

        while evaluations < self.budget:
            real_population, fitness = self._evaluate_population(func, bounds)
            self._update_best(real_population, fitness)
            evaluations += self.population_size
            self._dynamic_population_size(evaluations)
            self._adjust_mutation_rate(evaluations)
            self._adaptive_mutation()

        return self.best_solution