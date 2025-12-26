import numpy as np

class EAPDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9  # initial crossover rate
        self.pop = None
        self.bounds = None

    def __call__(self, func):
        # Initialize the population randomly within the bounds
        self.bounds = (func.bounds.lb, func.bounds.ub)
        lb, ub = self.bounds
        self.pop = lb + (ub - lb) * np.random.rand(self.initial_population_size, self.dim)
        fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.initial_population_size
        current_population_size = self.initial_population_size

        while evaluations < self.budget:
            new_pop = np.empty_like(self.pop)
            for i in range(current_population_size):
                # Select three distinct individuals randomly
                indices = np.random.choice([idx for idx in range(current_population_size) if idx != i], 3, replace=False)
                r1, r2, r3 = self.pop[indices]

                # Mutation and crossover
                dynamic_F = self.F * (1 - evaluations / self.budget)
                mutant = r1 + dynamic_F * (r2 - r3)
                mutant = np.clip(mutant, lb, ub)

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
            self.F = np.clip(0.5 + 0.3 * np.random.randn(), 0, 1)
            self.CR = np.clip(0.5 + 0.2 * np.random.randn(), 0, 1)

            # Dynamic population size adjustment
            if evaluations < self.budget * 0.5:
                current_population_size = min(current_population_size + 1, self.initial_population_size * 2)
            else:
                current_population_size = max(current_population_size - 1, self.initial_population_size // 2)

        # Return the best solution found
        best_index = np.argmin(fitness)
        return self.pop[best_index]