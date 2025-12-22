import numpy as np

class QSA_DE:
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

    def _differential_mutation(self, population, bounds, F=0.5, CR=0.9):
        mutated_population = np.empty_like(population)
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[indices]
            mutant = x1 + F * (x2 - x3)
            mutant = np.clip(mutant, 0, 1)
            trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
            trial_real = bounds.lb + trial * (bounds.ub - bounds.lb)
            if func(trial_real) < func(population[i]):
                mutated_population[i] = trial
            else:
                mutated_population[i] = population[i]
        return mutated_population

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0

        while evaluations < self.budget:
            real_population, fitness = self._evaluate_population(func, bounds)
            self._update_best(real_population, fitness)
            evaluations += self.population_size
            self.q_population = self._differential_mutation(self.q_population, bounds)

        return self.best_solution