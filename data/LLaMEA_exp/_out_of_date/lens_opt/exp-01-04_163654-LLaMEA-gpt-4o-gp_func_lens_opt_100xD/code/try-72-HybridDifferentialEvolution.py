import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 25  # Increased population size
        self.F = 0.6  # Modified mutation factor
        self.CR = 0.8  # Modified crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                if np.random.rand() < 0.5:  # Adaptive mutation strategy
                    a = population[np.argmin(fitness)]  # Use best solution for mutation
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
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

            # Random Immigrant Strategy
            if num_evaluations + self.pop_size // 5 < self.budget:  # Ensure budget is respected
                num_new = self.pop_size // 5
                new_immigrants = np.random.uniform(lb, ub, (num_new, self.dim))
                for j in range(num_new):
                    new_population[j] = new_immigrants[j]
                    fitness[j] = func(new_immigrants[j])
                    num_evaluations += 1

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]