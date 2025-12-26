import numpy as np

class EAPDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9  # initial crossover rate
        self.pop = None
        self.bounds = None

    def __call__(self, func):
        # Initialize the population randomly within the bounds
        self.bounds = (func.bounds.lb, func.bounds.ub)
        lb, ub = self.bounds
        self.pop = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.population_size

        while evaluations < self.budget:
            new_pop = np.empty_like(self.pop)
            for i in range(self.population_size):
                # Select three distinct individuals randomly
                indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                r1, r2, r3 = self.pop[indices]

                # Mutation and crossover
                mutant = r1 + self.F * (r2 - r3)
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

                # Local search intensification
                if evaluations < self.budget and np.random.rand() < 0.1:
                    local_step = 0.05 * (ub - lb) * np.random.randn(self.dim)
                    local_candidate = np.clip(trial + local_step, lb, ub)
                    local_fitness = func(local_candidate)
                    evaluations += 1
                    if local_fitness < trial_fitness:
                        new_pop[i] = local_candidate
                        fitness[i] = local_fitness

            self.pop = new_pop

            # Adaptive adjustments of F and CR
            self.F = np.clip(0.5 + 0.3 * np.random.randn(), 0.1, 0.9)  # Ensuring F stays within a reasonable range
            self.CR = np.clip(0.5 + 0.2 * np.random.randn(), 0.1, 1)  # More exploration in CR adaptation

        # Return the best solution found
        best_index = np.argmin(fitness)
        return self.pop[best_index]