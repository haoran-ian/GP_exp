import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(10 * dim, budget // 10)  # Dynamic population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.8  # Differential weight
        self.recombination_rate = 0.9  # Crossover probability

    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                spatial_factor = np.random.uniform(0.5, 1.0, self.dim)  # New line
                mutant = np.clip(a + self.mutation_factor * (b - c) * spatial_factor, self.lower_bound, self.upper_bound)  # Modified line

                # Crossover
                cross_points = np.random.rand(self.dim) < self.recombination_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]