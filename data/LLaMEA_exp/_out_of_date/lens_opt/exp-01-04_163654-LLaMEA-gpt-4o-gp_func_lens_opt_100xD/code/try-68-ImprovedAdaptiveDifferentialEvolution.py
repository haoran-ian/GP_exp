import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.base_F = 0.5
        self.base_CR = 0.9
        self.success_rate_threshold = 0.2
        self.mutation_decay = 0.995
        self.crossover_decay = 0.995

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        adaptive_F = self.base_F
        adaptive_CR = self.base_CR
        last_improvement = 0

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            success_count = 0

            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + adaptive_F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < adaptive_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    success_count += 1
                    last_improvement = num_evaluations

                if num_evaluations >= self.budget:
                    break

            population = new_population

            success_rate = success_count / self.pop_size
            if success_rate < self.success_rate_threshold:
                adaptive_F *= self.mutation_decay
                adaptive_CR *= self.crossover_decay
            else:
                adaptive_F = min(1.0, adaptive_F / self.mutation_decay)
                adaptive_CR = min(1.0, adaptive_CR / self.crossover_decay)

            # Reset mutation and crossover rates on stagnation
            if num_evaluations - last_improvement > self.pop_size * 2:
                adaptive_F = self.base_F
                adaptive_CR = self.base_CR

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]