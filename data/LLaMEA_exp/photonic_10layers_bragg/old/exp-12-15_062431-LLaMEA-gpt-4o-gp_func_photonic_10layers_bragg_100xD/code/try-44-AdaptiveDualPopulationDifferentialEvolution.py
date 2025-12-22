import numpy as np

class AdaptiveDualPopulationDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 12 * dim
        self.secondary_population_size = 6 * dim
        self.min_population_size = 4 * dim
        self.primary_population_size = self.initial_population_size
        self.eval_count = 0
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.alpha = 0.1
        self.diversity_threshold = 0.1

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        primary_population = np.random.rand(self.primary_population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        secondary_population = np.random.rand(self.secondary_population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        
        primary_fitness = np.array([func(ind) for ind in primary_population])
        secondary_fitness = np.array([func(ind) for ind in secondary_population])
        self.eval_count += self.primary_population_size + self.secondary_population_size

        while self.eval_count < self.budget:
            for population, fitness, size in [(primary_population, primary_fitness, self.primary_population_size), 
                                              (secondary_population, secondary_fitness, self.secondary_population_size)]:
                for i in range(size):
                    indices = np.random.choice([j for j in range(size) if j != i], 3, replace=False)
                    x1, x2, x3 = population[indices]
                    oscillating_factor = np.sin(2 * np.pi * self.eval_count / self.budget)
                    self_adaptive_mutation = self.alpha * np.random.randn(self.dim)
                    mutant = np.clip(x1 + self.mutation_factor * (x2 - x3) * oscillating_factor + self_adaptive_mutation, bounds[0], bounds[1])

                    cross_points = np.random.rand(self.dim) < self.crossover_rate
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])

                    f_trial = func(trial)
                    self.eval_count += 1
                    if f_trial < fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial

                if self.eval_count % (self.budget // 10) == 0 and size > self.min_population_size:
                    size = max(self.min_population_size, size // 2)
                    indices = np.argsort(fitness)[:size]
                    population[:] = population[indices]
                    fitness[:] = fitness[indices]

                diversity = np.std(population, axis=0).mean()
                if population is primary_population:
                    self.crossover_rate = 0.3 + 0.4 * np.sin(2 * np.pi * self.eval_count / self.budget)
                    self.mutation_factor = 0.5 + 0.3 * np.cos(2 * np.pi * (self.eval_count/self.budget) * (fitness.mean() / (fitness.min() + 1e-8)))
                    if diversity < self.diversity_threshold:
                        new_individuals = np.random.rand(self.initial_population_size - size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
                        population = np.vstack((population, new_individuals))
                        fitness = np.append(fitness, [func(ind) for ind in new_individuals])
                        self.eval_count += len(new_individuals)

        combined_population = np.vstack((primary_population, secondary_population))
        combined_fitness = np.append(primary_fitness, secondary_fitness)
        best_index = np.argmin(combined_fitness)
        return combined_population[best_index], combined_fitness[best_index]