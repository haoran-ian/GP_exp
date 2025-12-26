import numpy as np

class EAPDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.max_population_size = 20 * dim
        self.F = 0.5
        self.CR = 0.9
        self.pop = None
        self.bounds = None

    def __call__(self, func):
        # Initialize the population randomly within the bounds
        self.bounds = (func.bounds.lb, func.bounds.ub)
        lb, ub = self.bounds
        self.pop = lb + (ub - lb) * np.random.rand(self.initial_population_size, self.dim)
        fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.initial_population_size

        while evaluations < self.budget:
            new_pop = np.empty_like(self.pop)
            for i in range(len(self.pop)):
                # Dynamic population size adjustment
                if evaluations > self.budget / 2 and len(self.pop) < self.max_population_size:
                    self.pop = np.vstack((self.pop, lb + (ub - lb) * np.random.rand(dim)))

                # Select three distinct individuals randomly
                indices = np.random.choice([idx for idx in range(len(self.pop)) if idx != i], 3, replace=False)
                r1, r2, r3 = self.pop[indices]

                # Mutation with adaptive F
                F_dynamic = self.F * (1 - (evaluations / self.budget))
                mutant = r1 + F_dynamic * (r2 - r3)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])

                # Evaluate trial and perform selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_pop[i] = self.pop[i]

            self.pop = new_pop

            # Adaptive adjustments of F and CR
            self.F = np.clip(self.F + 0.05 * (np.mean(fitness) - np.min(fitness)), 0, 1)
            self.CR = np.clip(self.CR + 0.1 * np.random.randn(), 0, 1)

        # Return the best solution found
        best_index = np.argmin(fitness)
        return self.pop[best_index]