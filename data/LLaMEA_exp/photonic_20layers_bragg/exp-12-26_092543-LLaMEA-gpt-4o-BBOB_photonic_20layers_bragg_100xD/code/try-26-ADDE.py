import numpy as np

class ADDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
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
        successful_mutations = []

        while evaluations < self.budget:
            new_pop = np.empty_like(self.pop)
            for i in range(self.population_size):
                # Select three distinct individuals randomly
                indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                r1, r2, r3 = self.pop[indices]

                # Mutation and crossover with self-adaptive F and CR
                F_local = np.clip(self.F + 0.1 * np.random.randn(), 0, 1)
                CR_local = np.clip(self.CR + 0.05 * np.random.randn(), 0, 1)
                mutant = r1 + F_local * (r2 - r3)
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

            # Adaptive adjustments of F and CR based on success and adaptive learning
            if successful_mutations:
                self.F = np.mean([x[0] for x in successful_mutations])
                self.CR = np.mean([x[1] for x in successful_mutations])
                successful_mutations.clear()

            # Introduce dynamic population resizing
            if evaluations < self.budget:
                diversity = np.std(self.pop)
                if diversity < 0.1 and self.population_size > self.initial_population_size / 2:
                    self.population_size = max(self.initial_population_size // 2, 5)
                    indices = np.argsort(fitness)[:self.population_size]
                    self.pop = self.pop[indices]
                    fitness = fitness[indices]
                elif diversity > 0.5 and self.population_size < self.initial_population_size * 2:
                    extra_pop_size = min(self.initial_population_size // 2, self.budget - evaluations)
                    extra_pop = lb + (ub - lb) * np.random.rand(extra_pop_size, self.dim)
                    extra_fitness = np.array([func(ind) for ind in extra_pop])
                    evaluations += extra_pop_size
                    self.pop = np.vstack((self.pop, extra_pop))
                    fitness = np.append(fitness, extra_fitness)
                    self.population_size += extra_pop_size

        # Return the best solution found
        best_index = np.argmin(fitness)
        return self.pop[best_index]