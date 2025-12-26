import numpy as np

class DynamicADE:
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
        history = []

        while evaluations < self.budget:
            new_pop = np.empty_like(self.pop)
            for i in range(self.population_size):
                # Stochastic selection based on fitness ranking
                probabilities = np.exp(-np.argsort(np.argsort(fitness)))
                probabilities /= probabilities.sum()
                indices = np.random.choice(self.population_size, 3, replace=False, p=probabilities)
                r1, r2, r3 = self.pop[indices]

                # Mutation and crossover with adaptive F and CR
                F_local = np.clip(self.F + 0.1 * np.random.randn(), 0.4, 0.9)
                CR_local = np.clip(self.CR + 0.05 * np.random.randn(), 0.8, 1.0)
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
                    history.append(evaluations)
                else:
                    new_pop[i] = self.pop[i]

            self.pop = new_pop

            # Adaptive adjustments of F and CR based on success history
            if successful_mutations:
                self.F = np.mean([x[0] for x in successful_mutations])
                self.CR = np.mean([x[1] for x in successful_mutations])
                successful_mutations.clear()
                # Adjust population size based on success
                success_rate = len(history) / self.population_size
                if success_rate > 0.2:
                    self.population_size = min(20 * self.dim, int(self.population_size * 1.2))
                else:
                    self.population_size = max(5 * self.dim, int(self.population_size * 0.8))
                history.clear()

            # Preserve diversity by introducing new random individuals
            if evaluations < self.budget:
                num_randoms = int(0.1 * self.population_size)
                random_indices = np.random.choice(self.population_size, num_randoms, replace=False)
                for ri in random_indices:
                    self.pop[ri] = lb + (ub - lb) * np.random.rand(self.dim)
                    fitness[ri] = func(self.pop[ri])
                    evaluations += 1

        # Return the best solution found
        best_index = np.argmin(fitness)
        return self.pop[best_index]