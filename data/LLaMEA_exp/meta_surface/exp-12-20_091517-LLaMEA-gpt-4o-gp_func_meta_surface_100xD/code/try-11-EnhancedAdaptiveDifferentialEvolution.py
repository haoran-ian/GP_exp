import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.pop = None
        self.fitness = None
        self.bounds = None
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover rate
        self.successful_mutations = []

    def __call__(self, func):
        # Initialization
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.population_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Select three distinct vectors
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutation
                mutant = np.clip(self.pop[a] + self.F * (self.pop[b] - self.pop[c]), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.pop[i] = trial
                    self.successful_mutations.append((self.F, self.CR))

                if evaluations >= self.budget:
                    break

            # Learning-Based Dynamic Adjustment of F and CR
            if self.successful_mutations:
                self.F = np.clip(np.mean([f for f, cr in self.successful_mutations]) + np.random.uniform(-0.05, 0.05), 0.1, 0.9)
                self.CR = np.clip(np.mean([cr for f, cr in self.successful_mutations]) + np.random.uniform(-0.05, 0.05), 0.1, 1.0)
                self.successful_mutations.clear()

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]