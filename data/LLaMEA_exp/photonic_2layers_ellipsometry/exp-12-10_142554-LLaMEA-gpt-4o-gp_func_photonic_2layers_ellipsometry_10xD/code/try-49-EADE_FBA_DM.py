import numpy as np

class EADE_FBA_DM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_mutation_factor = 0.7
        self.initial_crossover_rate = 0.7
        self.mutation_factor = self.initial_mutation_factor
        self.crossover_rate = self.initial_crossover_rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                
                # Dynamic adaptation and differential mutation
                if eval_count % (self.population_size * 2) == 0:
                    diversity = np.std(pop, axis=0).mean() / self.dim
                    self.mutation_factor = 0.5 + 0.5 * diversity
                    self.crossover_rate = 0.2 + 0.6 * (1 - diversity)
                    if diversity < 0.1:  # Introduce differential mutation when diversity is low
                        a, b, c = pop[np.random.choice(self.population_size, 3, replace=False)]
                        pop[i] = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                        fitness[i] = func(pop[i])
                        eval_count += 1

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]