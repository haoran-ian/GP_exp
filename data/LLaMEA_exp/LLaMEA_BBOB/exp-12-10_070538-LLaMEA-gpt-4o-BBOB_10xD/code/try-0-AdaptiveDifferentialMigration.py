import numpy as np

class AdaptiveDifferentialMigration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 + 2 * dim  # Population size scaling with dimension
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.best = None

    def __call__(self, func):
        eval_count = 0
        fitness = np.apply_along_axis(func, 1, self.population)
        eval_count += self.population_size
        self.best = self.population[np.argmin(fitness)]

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                # Select three random indices different from i
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                # Differential mutation
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness

                # Update the best solution
                if trial_fitness < func(self.best):
                    self.best = trial

        return self.best