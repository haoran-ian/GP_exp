import numpy as np

class AQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.q_population = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')

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
        ranked_fitness = np.argsort(np.argsort(self.q_population.sum(axis=1)))
        mutation_strength = 0.01 / (1 + ranked_fitness)
        adapt_factor = np.random.rand(*self.q_population.shape)
        self.q_population += mutation_strength.reshape(-1, 1) * (adapt_factor - 0.5)
        self.q_population = np.clip(self.q_population, 0, 1)

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0

        while evaluations < self.budget:
            real_population, fitness = self._evaluate_population(func, bounds)
            self._update_best(real_population, fitness)
            evaluations += self.population_size
            self._adaptive_mutation()

        return self.best_solution