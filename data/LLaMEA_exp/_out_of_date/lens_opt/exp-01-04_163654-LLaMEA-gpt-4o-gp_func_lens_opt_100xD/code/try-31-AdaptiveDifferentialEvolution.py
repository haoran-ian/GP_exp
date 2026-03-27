import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.initial_F = 0.5
        self.initial_CR = 0.9
        self.alpha = 0.95  # Adaptive reduction factor for F and CR

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        current_F = self.initial_F
        current_CR = self.initial_CR

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + current_F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < current_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                if num_evaluations >= self.budget:
                    break

            # Adapt F and CR based on the improvement in fitness
            improvement_rate = np.sum(fitness < np.array([func(ind) for ind in population])) / self.pop_size
            current_F *= (1 - self.alpha * improvement_rate)
            current_CR *= (1 - self.alpha * improvement_rate)

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]