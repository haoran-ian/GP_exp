import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 12 * dim  # Changed initial population size
        self.min_population_size = 3 * dim  # Changed minimum population size
        self.population_size = self.initial_population_size
        self.eval_count = 0
        self.crossover_rate = 0.85  # Adjusted crossover rate
        self.mutation_factor = 0.7  # Adjusted mutation factor
        self.alpha = 0.15  # Changed self-adaptive mutation strategy parameter

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.rand(self.population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            diversity = np.std(population, axis=0).mean()  # Calculate diversity
            for i in range(self.population_size):
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
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

            if self.eval_count % (self.budget // 8) == 0 and self.population_size > self.min_population_size:
                self.population_size = max(self.min_population_size, int(self.population_size * (1 + diversity)))  # Adjusted population reduction
                indices = np.argsort(fitness)[:self.population_size]
                population = population[indices]
                fitness = fitness[indices]

            self.crossover_rate = 0.4 + 0.2 * np.sin(2 * np.pi * self.eval_count / self.budget)
            self.mutation_factor = 0.6 + 0.25 * np.cos(2 * np.pi * (self.eval_count/self.budget) * (fitness.mean() / fitness.min()))

            elite_size = max(1, self.population_size // 8)  # Adjusted elite size
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_population = population[elite_indices]
            population = np.vstack((elite_population, population))
            fitness = np.append(fitness[elite_indices], fitness)

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]