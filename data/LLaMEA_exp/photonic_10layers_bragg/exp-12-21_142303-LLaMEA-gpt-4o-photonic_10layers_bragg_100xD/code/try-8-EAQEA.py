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
        self.elite_fraction = 0.1  # Fraction of elite individuals to preserve

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

    def _preserve_elites(self, real_population, fitness):
        elite_count = max(1, int(self.elite_fraction * self.population_size))
        elite_indices = np.argsort(fitness)[:elite_count]
        self.q_population[:elite_count] = real_population[elite_indices]

    def _dynamic_population_resizing(self, evaluations):
        # Gradually increase/decrease population size based on budget consumption
        factor = (1 - evaluations / self.budget)
        self.population_size = max(5, int(self.initial_population_size * factor))
        self.q_population = np.resize(self.q_population, (self.population_size, self.dim))
        self.q_population = np.clip(self.q_population, 0, 1)

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0

        while evaluations < self.budget:
            real_population, fitness = self._evaluate_population(func, bounds)
            self._update_best(real_population, fitness)
            self._preserve_elites(real_population, fitness)
            evaluations += self.population_size
            self._adaptive_mutation()
            self._dynamic_population_resizing(evaluations)

        return self.best_solution