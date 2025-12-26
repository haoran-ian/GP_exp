import numpy as np

class AEDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9  # initial crossover rate
        self.pop = None
        self.bounds = None
        self.mutation_strategies = ['rand', 'best', 'current-to-best']

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        lb, ub = self.bounds
        self.pop = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.population_size
        successful_mutations = []

        while evaluations < self.budget:
            new_pop = np.empty_like(self.pop)
            for i in range(self.population_size):
                strategy = np.random.choice(self.mutation_strategies)
                if strategy == 'rand':
                    indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                    r1, r2, r3 = self.pop[indices]
                    mutant = r1 + self.F * (r2 - r3)
                elif strategy == 'best':
                    best_idx = np.argmin(fitness)
                    indices = np.random.choice([idx for idx in range(self.population_size) if idx != i and idx != best_idx], 2, replace=False)
                    r2, r3 = self.pop[indices]
                    mutant = self.pop[best_idx] + self.F * (r2 - r3)
                elif strategy == 'current-to-best':
                    best_idx = np.argmin(fitness)
                    indices = np.random.choice([idx for idx in range(self.population_size) if idx != i and idx != best_idx], 1, replace=False)
                    r1 = self.pop[indices[0]]
                    mutant = self.pop[i] + self.F * (self.pop[best_idx] - self.pop[i]) + self.F * (r1 - self.pop[i])

                mutant = np.clip(mutant, lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                    successful_mutations.append((self.F, self.CR))
                else:
                    new_pop[i] = self.pop[i]

            self.pop = new_pop

            if successful_mutations:
                self.F = np.clip(np.mean([x[0] for x in successful_mutations]) + 0.1 * np.random.randn(), 0.1, 0.9)
                self.CR = np.clip(np.mean([x[1] for x in successful_mutations]) + 0.05 * np.random.randn(), 0.1, 0.9)
                successful_mutations.clear()
                self.population_size = max(5 * self.dim, int(10 * self.dim * len(successful_mutations) / self.population_size))

            if evaluations < self.budget and np.random.rand() < 0.1:
                rand_index = np.random.randint(self.population_size)
                self.pop[rand_index] = lb + (ub - lb) * np.random.rand(self.dim)
                fitness[rand_index] = func(self.pop[rand_index])
                evaluations += 1

        best_index = np.argmin(fitness)
        return self.pop[best_index]