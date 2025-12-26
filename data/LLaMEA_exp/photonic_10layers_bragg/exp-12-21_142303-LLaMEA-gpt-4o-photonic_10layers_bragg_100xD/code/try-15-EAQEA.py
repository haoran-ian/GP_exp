import numpy as np

class EAQEA:
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
        mutation_strength = np.abs(np.random.normal(0, 0.05, self.q_population.shape))
        adapt_factor = np.random.rand(*self.q_population.shape)
        self.q_population += mutation_strength * (adapt_factor - 0.5)
        self.q_population = np.clip(self.q_population, 0, 1)

    def _levy_flight(self, L):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / abs(v) ** (1 / beta)
        return step

    def _global_search(self, bounds):
        for _ in range(self.population_size):
            random_individual = np.random.uniform(bounds.lb, bounds.ub, self.dim)
            step_size = self._levy_flight(self.dim)
            random_individual += 0.01 * step_size * (random_individual - self.best_solution)
            random_individual = np.clip(random_individual, bounds.lb, bounds.ub)
            yield random_individual

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0

        while evaluations < self.budget:
            real_population, fitness = self._evaluate_population(func, bounds)
            self._update_best(real_population, fitness)
            evaluations += self.population_size

            # Hybrid exploration-exploitation strategy
            if evaluations < self.budget // 2:
                self._adaptive_mutation()
            else:
                for individual in self._global_search(bounds):
                    ind_fitness = func(individual)
                    if ind_fitness < self.best_fitness:
                        self.best_fitness = ind_fitness
                        self.best_solution = individual
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

        return self.best_solution