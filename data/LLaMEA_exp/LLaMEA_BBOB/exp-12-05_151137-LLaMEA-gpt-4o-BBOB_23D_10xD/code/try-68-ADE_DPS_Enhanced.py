import numpy as np

class ADE_DPS_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5 + np.random.rand() * 0.5  # Adaptive differential weight
        self.CR = np.full(self.population_size, 0.7) + np.random.rand(self.population_size) * 0.3  # Individual adaptive crossover probabilities

    def __call__(self, func):
        evals = 0
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals += self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                # Adaptive mutation strategy based on fitness ranking
                indices = np.random.choice(self.population_size, 3, replace=False)
                x_1, x_2, x_3 = population[indices]
                if fitness[i] > np.median(fitness):
                    mutant = np.clip(x_1 + self.F * (x_2 - x_3), self.lower_bound, self.upper_bound)
                else:
                    mutant = np.clip(x_1 + 0.5 * self.F * (x_2 - x_3), self.lower_bound, self.upper_bound)

                # Enhanced Crossover
                cross_points = np.random.rand(self.dim) < self.CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            if evals < self.budget * 0.5:
                self.population_size = min(self.population_size * 2, self.budget - evals + 1)
            else:
                self.population_size = max(5, self.population_size // 2)
            if len(population) != self.population_size:
                if len(population) < self.population_size:
                    additional_pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size - len(population), self.dim))
                    additional_fitness = np.array([func(ind) for ind in additional_pop])
                    evals += len(additional_pop)
                    population = np.vstack([population, additional_pop])
                    fitness = np.hstack([fitness, additional_fitness])
                else:
                    selected = np.argsort(fitness)[:self.population_size]
                    population = population[selected]
                    fitness = fitness[selected]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]