import numpy as np

class AdvancedEvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 12 * dim
        self.min_population_size = 6 * dim
        self.population_size = self.initial_population_size
        self.eval_count = 0
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.alpha = 0.15
        self.diversity_threshold = 0.15
        self.memory_size = 5
        self.successful_mutations = []

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.rand(self.population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                adaptive_factor = np.var(fitness) / (np.mean(fitness) + 1e-8)
                self_adaptive_mutation = self.alpha * np.random.randn(self.dim)
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3) * adaptive_factor + self_adaptive_mutation, bounds[0], bounds[1])

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    self.successful_mutations.append(self.mutation_factor)

            if len(self.successful_mutations) > self.memory_size:
                self.successful_mutations.pop(0)
            self.mutation_factor = 0.5 + 0.3 * np.mean(self.successful_mutations) if self.successful_mutations else 0.5

            if self.eval_count % (self.budget // 8) == 0 and self.population_size > self.min_population_size:
                self.population_size = max(self.min_population_size, self.population_size // 2)
                indices = np.argsort(fitness)[:self.population_size]
                population = population[indices]
                fitness = fitness[indices]

            population_diversity = np.std(population, axis=0).mean()

            if population_diversity < self.diversity_threshold:
                new_individuals = np.random.rand(self.initial_population_size - self.population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
                population = np.vstack((population, new_individuals))
                fitness = np.append(fitness, [func(ind) for ind in new_individuals])
                self.eval_count += len(new_individuals)

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]