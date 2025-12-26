import numpy as np

class OEAPDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.pop = None
        self.bounds = None

    def __call__(self, func):
        # Initialize the population randomly within the bounds
        self.bounds = (func.bounds.lb, func.bounds.ub)
        lb, ub = self.bounds
        self.pop = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.population_size
        successful_mutations = []

        while evaluations < self.budget:
            new_pop = np.empty_like(self.pop)
            for i in range(self.population_size):
                # Select three distinct individuals randomly
                indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                r1, r2, r3 = self.pop[indices]

                # Mutation and crossover with adaptive F and CR based on success rate
                F_local = np.clip(self.F + 0.1 * np.random.randn(), 0, 1)
                CR_local = np.clip(self.CR + 0.05 * np.random.randn(), 0, 1)
                if np.random.rand() < 0.5:
                    mutant = r1 + F_local * (r2 - r3)
                else:
                    mutant = r1 - F_local * (r2 - r3)
                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < CR_local
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])

                # Evaluate trial and perform selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                    successful_mutations.append((F_local, CR_local))
                else:
                    new_pop[i] = self.pop[i]

            self.pop = new_pop

            # Update F and CR based on successful history with learning rate adaptation
            if successful_mutations:
                self.F = 0.9 * self.F + 0.1 * np.mean([x[0] for x in successful_mutations])
                self.CR = 0.9 * self.CR + 0.1 * np.mean([x[1] for x in successful_mutations])
                successful_mutations.clear()

            # Dynamic diversity preservation
            if evaluations < self.budget and np.random.rand() < 0.2:
                diversity_indices = np.random.choice(self.population_size, size=self.population_size // 5, replace=False)
                for idx in diversity_indices:
                    self.pop[idx] = lb + (ub - lb) * np.random.rand(self.dim)
                    fitness[idx] = func(self.pop[idx])
                    evaluations += 1

        # Return the best solution found
        best_index = np.argmin(fitness)
        return self.pop[best_index]