import numpy as np

class ImprovedEAPDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9  # initial crossover rate
        self.pop = None
        self.bounds = None
        self.evaluations = 0
        self.successful_mutations = []

    def __call__(self, func):
        # Initialize the population randomly within the bounds
        self.bounds = (func.bounds.lb, func.bounds.ub)
        lb, ub = self.bounds
        self.pop = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in self.pop])
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            new_pop = np.empty_like(self.pop)

            # Adaptive mutation scaling based on generation progress
            generation_progress = self.evaluations / self.budget
            dynamic_F = self.F * (1 - generation_progress)

            for i in range(self.population_size):
                # Select three distinct individuals randomly
                indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                r1, r2, r3 = self.pop[indices]

                # Apply mutation and crossover with dynamic scaling
                F_local = np.clip(dynamic_F + 0.1 * np.random.randn(), 0, 1)
                CR_local = np.clip(self.CR + 0.05 * np.random.randn(), 0, 1)
                mutant = r1 + F_local * (r2 - r3)
                mutant = np.clip(mutant, lb, ub)

                # Adaptive learning of crossover strategies
                if np.random.rand() < 0.5:
                    trial = np.where(np.random.rand(self.dim) < CR_local, mutant, self.pop[i])
                else:
                    trial = mutant  # Full trial as mutant

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                    self.successful_mutations.append((F_local, CR_local))
                else:
                    new_pop[i] = self.pop[i]

            self.pop = new_pop

            # Update F and CR based on successful trials
            if self.successful_mutations:
                self.F = np.mean([x[0] for x in self.successful_mutations])
                self.CR = np.mean([x[1] for x in self.successful_mutations])
                self.successful_mutations.clear()

            # Occasionally introduce random individuals
            if self.evaluations < self.budget and np.random.rand() < 0.1:
                rand_index = np.random.randint(self.population_size)
                self.pop[rand_index] = lb + (ub - lb) * np.random.rand(self.dim)
                fitness[rand_index] = func(self.pop[rand_index])
                self.evaluations += 1

        # Return the best solution found
        best_index = np.argmin(fitness)
        return self.pop[best_index]