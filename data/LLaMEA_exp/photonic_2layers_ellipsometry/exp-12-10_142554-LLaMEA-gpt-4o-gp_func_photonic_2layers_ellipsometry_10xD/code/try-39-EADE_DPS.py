import numpy as np

class EADE_DPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_population_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = pop_size

        while eval_count < self.budget:
            for i in range(pop_size):
                indices = list(range(0, i)) + list(range(i+1, pop_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

            # Dynamic adaptation of factors and population resizing
            if eval_count % (pop_size * 2) == 0:
                diversity = np.std(fitness) / (np.mean(fitness) + 1e-9)
                self.mutation_factor = 0.3 + 0.7 * diversity
                self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

                # Adjust population size based on convergence
                if diversity < 0.1:
                    pop_size = max(int(pop_size * 0.9), self.dim)
                else:
                    pop_size = min(int(pop_size * 1.1), self.initial_population_size)

                pop = pop[:pop_size]
                fitness = fitness[:pop_size]

            if eval_count >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]