import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F_base = 0.5
        self.CR_base = 0.9
        self.scale_factor_range = (0.4, 0.9)
        self.mutation_strategies = ['rand/1', 'best/1', 'current-to-best/1']
        self.strategy_probabilities = np.ones(len(self.mutation_strategies)) / len(self.mutation_strategies)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                strategy = np.random.choice(self.mutation_strategies, p=self.strategy_probabilities)
                F = np.random.uniform(*self.scale_factor_range)
                CR = self.CR_base

                if strategy == 'rand/1':
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), lb, ub)
                elif strategy == 'best/1':
                    best_idx = np.argmin(fitness)
                    a, b = population[np.random.choice(idxs, 2, replace=False)]
                    mutant = np.clip(population[best_idx] + F * (a - b), lb, ub)
                elif strategy == 'current-to-best/1':
                    best_idx = np.argmin(fitness)
                    a, b = population[np.random.choice(idxs, 2, replace=False)]
                    mutant = np.clip(population[i] + F * (population[best_idx] - population[i]) + F * (a - b), lb, ub)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.strategy_probabilities[self.mutation_strategies.index(strategy)] += 0.1
                else:
                    self.strategy_probabilities[self.mutation_strategies.index(strategy)] -= 0.1

                self.strategy_probabilities = np.clip(self.strategy_probabilities, 0.01, None)
                self.strategy_probabilities /= np.sum(self.strategy_probabilities)

                if num_evaluations >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]