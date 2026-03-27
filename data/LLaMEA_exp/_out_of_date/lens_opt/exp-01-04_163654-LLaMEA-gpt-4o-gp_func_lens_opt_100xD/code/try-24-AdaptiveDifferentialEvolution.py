import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F = 0.5
        self.CR = 0.9
        self.mutation_strategy = 'rand/1'

    def adaptive_parameters(self, iteration, max_iterations):
        # Self-adaptive control parameters
        self.F = 0.5 + 0.3 * np.sin((np.pi * iteration) / max_iterations)
        self.CR = 0.8 + 0.2 * np.cos((np.pi * iteration) / max_iterations)

    def mutate(self, a, b, c):
        if self.mutation_strategy == 'rand/1':
            return a + self.F * (b - c)
        elif self.mutation_strategy == 'best/1':
            return self.best + self.F * (b - c)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size
        max_iterations = (self.budget - self.pop_size) // self.pop_size
        iteration = 0

        while num_evaluations < self.budget:
            self.best = population[np.argmin(fitness)]
            new_population = np.copy(population)
            self.adaptive_parameters(iteration, max_iterations)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(self.mutate(a, b, c), lb, ub)
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
            population = new_population
            iteration += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]