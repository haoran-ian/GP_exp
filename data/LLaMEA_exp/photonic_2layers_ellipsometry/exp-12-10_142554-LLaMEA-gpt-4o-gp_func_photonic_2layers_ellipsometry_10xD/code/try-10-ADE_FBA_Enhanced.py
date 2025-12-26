import numpy as np

class ADE_FBA_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.population_size = self.initial_population_size
        self.success_count = 0  # Track successful trials

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
                    self.success_count += 1  # Increment on success

                # Dynamic adaptation based on success rate
                if eval_count % (self.population_size * 2) == 0:
                    success_rate = self.success_count / (self.population_size * 2)
                    self.mutation_factor = 0.4 + 0.6 * success_rate
                    self.crossover_rate = 0.2 + 0.7 * (1 - success_rate)
                    self.success_count = 0  # Reset success count

                # Linear population size reduction
                if eval_count % (self.initial_population_size * 10) == 0 and self.population_size > 4:
                    self.population_size = max(4, self.population_size - 1)
                    pop = pop[:self.population_size]
                    fitness = fitness[:self.population_size]

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]