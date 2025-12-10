import numpy as np

class EAD_VFBA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.previous_mutations = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            indices = np.arange(self.population_size)
            np.random.shuffle(indices)
            for i in range(self.population_size):
                idxs = indices[indices != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                mutant = 0.5 * mutant + 0.5 * self.previous_mutations[i]  # Use previous mutations
                
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    self.previous_mutations[i] = mutant  # Store successful mutation

                if eval_count % (self.population_size * 2) == 0:
                    diversity = np.mean(np.linalg.norm(pop[:, None] - pop, axis=2)) / self.dim
                    self.mutation_factor = 0.4 + 0.6 * diversity  # Adjust factors
                    self.crossover_rate = 0.2 + 0.7 * (1 - diversity)

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]