import numpy as np

class ADE_FBA_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor_min = 0.3
        self.mutation_factor_max = 0.9
        self.crossover_rate_min = 0.5
        self.crossover_rate_max = 1.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        rank = np.argsort(fitness)

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + np.random.uniform(self.mutation_factor_min, self.mutation_factor_max) * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < np.random.uniform(self.crossover_rate_min, self.crossover_rate_max)
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                rank = np.argsort(fitness)
                
                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]