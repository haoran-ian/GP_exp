import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(max(4, 10 * dim), budget // 2)
        self.pop = None
        self.fitness = None
        self.bounds = None
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover rate

    def __call__(self, func):
        # Initialization
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.initial_population_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.initial_population_size
        
        while evaluations < self.budget:
            population_size = len(self.pop)
            for i in range(population_size):
                # Select three distinct vectors
                candidates = list(range(population_size))
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

                if evaluations >= self.budget:
                    break

            # Dynamic adjustment of F and CR
            self.F = np.clip(self.F + np.random.uniform(-0.1, 0.1), 0.1, 0.9)
            self.CR = np.clip(self.CR + np.random.uniform(-0.1, 0.1), 0.1, 1.0)

            # Self-adaptive population size control
            if evaluations < self.budget / 2:
                new_size = int(0.9 * population_size)
            else:
                new_size = int(1.1 * population_size)
            new_size = max(4, min(new_size, self.budget - evaluations))
            
            # Adjust population size
            if new_size < population_size:
                sort_idx = np.argsort(self.fitness)
                self.pop = self.pop[sort_idx[:new_size]]
                self.fitness = self.fitness[sort_idx[:new_size]]
            elif new_size > population_size:
                additional_pop = np.random.rand(new_size - population_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
                additional_fitness = np.array([func(ind) for ind in additional_pop])
                evaluations += len(additional_pop)
                self.pop = np.vstack((self.pop, additional_pop))
                self.fitness = np.concatenate((self.fitness, additional_fitness))
        
        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]