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

    def __call__(self, func):
        evals = 0
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals += self.population_size
        best_fitness = np.min(fitness)
        best_solution = population[np.argmin(fitness)]
        
        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                
                F = 0.5 + 0.5 * (1 - evals / self.budget)  # Adaptive F
                indices = np.random.choice(self.population_size, 3, replace=False)
                x_1, x_2, x_3 = population[indices]
                mutant = np.clip(x_1 + F * (x_2 - x_3), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial

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

        return best_solution, best_fitness