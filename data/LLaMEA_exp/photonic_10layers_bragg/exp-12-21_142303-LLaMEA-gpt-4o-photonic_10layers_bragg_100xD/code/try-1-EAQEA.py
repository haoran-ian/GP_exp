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
        mutation_strength = np.abs(np.random.normal(0, 0.05, self.q_population.shape))
        adapt_factor = np.random.rand(*self.q_population.shape)
        self.q_population += mutation_strength * (adapt_factor - 0.5)
        self.q_population = np.clip(self.q_population, 0, 1)

    def _elitism_selection(self, real_population, fitness):
        elite_count = max(1, self.population_size // 10)
        elite_indices = fitness.argsort()[:elite_count]
        elite_population = real_population[elite_indices]
        self.q_population[:elite_count] = elite_population

    def _dynamic_population_resize(self, evaluations):
        if evaluations < self.budget // 2:
            self.population_size = min(self.initial_population_size * 2, self.population_size + 5)
        else:
            self.population_size = max(self.initial_population_size // 2, self.population_size - 5)
        self.q_population = np.resize(self.q_population, (self.population_size, self.dim))

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0

        while evaluations < self.budget:
            real_population, fitness = self._evaluate_population(func, bounds)
            self._update_best(real_population, fitness)
            self._elitism_selection(real_population, fitness)
            evaluations += self.population_size
            self._adaptive_mutation()
            self._dynamic_population_resize(evaluations)

        return self.best_solution