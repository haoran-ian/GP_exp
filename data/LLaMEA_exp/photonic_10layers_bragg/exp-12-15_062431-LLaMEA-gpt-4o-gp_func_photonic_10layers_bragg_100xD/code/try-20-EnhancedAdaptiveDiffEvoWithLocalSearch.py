import numpy as np

class EnhancedAdaptiveDiffEvoWithLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.min_population_size = 4 * dim
        self.eval_count = 0
        # Dynamic control parameters
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.alpha = 0.1
        
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.rand(self.initial_population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.initial_population_size
        
        while self.eval_count < self.budget:
            for i in range(len(population)):
                indices = np.random.choice([j for j in range(len(population)) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                oscillating_factor = np.sin(2 * np.pi * self.eval_count / self.budget)
                local_search = np.random.normal(0, 0.1, self.dim)
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3) * oscillating_factor + self.alpha * local_search, bounds[0], bounds[1])

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

            # Update population size adaptively
            if self.eval_count % (self.budget // 10) == 0 and len(population) > self.min_population_size:
                self.population_size = max(self.min_population_size, len(population) // 2)
                indices = np.argsort(fitness)[:self.population_size]
                population = population[indices]
                fitness = fitness[indices]

            # Dynamic parameter adaptation
            self.crossover_rate = 0.5 + 0.4 * np.sin(2 * np.pi * self.eval_count / self.budget)
            self.mutation_factor = 0.5 + 0.4 * np.cos(2 * np.pi * (self.eval_count/self.budget) * (fitness.mean() / fitness.min()))

            # Maintain elite population
            elite_size = max(1, len(population) // 10)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_population = population[elite_indices]
            population = np.vstack((elite_population, population))
            fitness = np.append(fitness[elite_indices], fitness)

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]