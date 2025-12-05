import numpy as np

class ADE_DPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim  # Start with a reasonably sized population
        self.F = 0.9  # Differential weight, increased for better exploration
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        evals = 0
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals += self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x_1, x_2, x_3 = population[indices]
                mutant = np.clip(x_1 + self.F * (x_2 - x_3 + x_1 - population[i]), self.lower_bound, self.upper_bound)  # Modified mutation strategy

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Adaptive population sizing
            if evals < self.budget * 0.5:
                self.population_size = min(self.population_size * 2, self.budget - evals + 1)
            else:
                self.population_size = max(5, self.population_size // 2)
            # Adjust population size
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

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]