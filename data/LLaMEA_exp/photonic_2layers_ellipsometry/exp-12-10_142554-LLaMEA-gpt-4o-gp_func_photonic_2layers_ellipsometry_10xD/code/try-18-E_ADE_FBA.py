import numpy as np

class E_ADE_FBA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.restart_threshold = 0.2  # Threshold for adaptive restart

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Diversity-aware mutation strategy
                diversity = np.std(pop, axis=0) / (ub - lb)
                self.mutation_factor = 0.3 + 0.4 * np.mean(diversity)
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

            # Adaptive restart mechanism
            if eval_count % (self.population_size * 2) == 0:
                mean_fitness = np.mean(fitness)
                worst_fitness = np.max(fitness)
                if (worst_fitness - mean_fitness) / (np.abs(mean_fitness) + 1e-12) < self.restart_threshold:
                    pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
                    fitness = np.array([func(ind) for ind in pop])
                    eval_count += self.population_size

                # Dynamic adaptation of crossover rate
                diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
                self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]