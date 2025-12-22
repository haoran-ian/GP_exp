import numpy as np

class EQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.q_population = np.random.uniform(0, 1, (self.initial_population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population_growth_rate = 1.05

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

    def _adaptive_crossover(self):
        self.crossover_rate = 0.5 + 0.5 * (self.budget - self.remaining_budget) / self.budget

    def _differential_evolution(self, real_population, bounds):
        trial_population = np.copy(real_population)
        for i in range(real_population.shape[0]):
            indices = [idx for idx in range(real_population.shape[0]) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = real_population[a] + self.mutation_factor * (real_population[b] - real_population[c])
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            crossover = np.random.rand(self.dim) < self.crossover_rate
            trial_population[i] = np.where(crossover, mutant, real_population[i])
        return trial_population

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0
        self.remaining_budget = self.budget

        while evaluations < self.budget:
            real_population, fitness = self._evaluate_population(func, bounds)
            self._update_best(real_population, fitness)
            evaluations += real_population.shape[0]
            self.remaining_budget -= real_population.shape[0]
            trial_population = self._differential_evolution(real_population, bounds)
            trial_fitness = np.array([func(ind) for ind in trial_population])
            
            for i in range(real_population.shape[0]):
                if trial_fitness[i] < fitness[i]:
                    fitness[i] = trial_fitness[i]
                    real_population[i] = trial_population[i]
            
            self._adaptive_mutation()
            self._adaptive_crossover()

            # Dynamic population size adjustment
            if self.remaining_budget > 0:
                new_population_size = int(self.q_population.shape[0] * self.population_growth_rate)
                if new_population_size > self.remaining_budget:
                    new_population_size = self.remaining_budget
                self.q_population = np.random.uniform(0, 1, (new_population_size, self.dim))

        return self.best_solution