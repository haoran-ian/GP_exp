import numpy as np

class EADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, 10 * dim // 2)  # Dynamic population size
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.success_mutation_factors = []  # Track successful mutation factors

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                F = self.mutation_factor if not self.success_mutation_factors else np.mean(self.success_mutation_factors)
                mutant = np.clip(a + F * (b - c), lb, ub)  # Use mean of successful mutations

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    self.success_mutation_factors.append(F)  # Store successful mutation factor

                if eval_count % (self.population_size * 2) == 0:
                    diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
                    self.mutation_factor = 0.5 + 0.5 * diversity  # Adjusted mutation factor range
                    self.crossover_rate = 0.1 + 0.8 * (1 - diversity)
                    if diversity < 0.1:  # Dynamic population size adjustment
                        self.population_size = min(2 * self.population_size, 10 * dim)

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]