import numpy as np

class ADE_DPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.7 + np.random.rand() * 0.3
        self.CR = np.full(self.population_size, 0.8) + np.random.rand(self.population_size) * 0.2
        self.local_search_period = max(1, self.budget // (5 * self.population_size))  # New

    def __call__(self, func):
        evals = 0
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals += self.population_size

        while evals < self.budget:
            if evals % self.local_search_period == 0:                         # New
                best_idx = np.argmin(fitness)
                local_candidate = population[best_idx] + np.random.uniform(-0.5, 0.5, self.dim)  # New
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)  # New
                local_fitness = func(local_candidate)                                           # New
                evals += 1                                                                      # New
                if local_fitness < fitness[best_idx]:                                           # New
                    population[best_idx] = local_candidate                                      # New
                    fitness[best_idx] = local_fitness                                           # New

            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                x_1, x_2, x_3 = population[indices]
                F_adaptive = self.F * (1 + (np.min(fitness) / fitness[i]))  # Updated
                mutant = np.clip(x_1 + F_adaptive * (x_2 - x_3), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

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