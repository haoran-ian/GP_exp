import numpy as np

class AdaptiveOscillationDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.min_population_size = 4 * dim
        self.population_size = self.initial_population_size
        self.eval_count = 0
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.alpha = 0.1
        self.diversity_threshold = 0.05

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.rand(self.population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        p_best_rate = 0.2

        while self.eval_count < self.budget:
            fitness_variance = np.var(fitness)
            for i in range(self.population_size):
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                oscillating_factor = np.sin(5 * np.pi * self.eval_count / self.budget)  # Reduced oscillation frequency
                
                self_adaptive_mutation = self.alpha * np.random.randn(self.dim) * (1 - self.eval_count / self.budget)
                
                p_best = population[np.argsort(fitness)[:max(1, int(p_best_rate * self.population_size))]][0]
                noise_scale = 0.05 * (1.0 - self.eval_count / self.budget)
                noise = np.random.randn(self.dim) * noise_scale
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3) * oscillating_factor +
                                 self_adaptive_mutation + 0.6 * (p_best - x1) + noise, bounds[0], bounds[1])
                
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

            if self.eval_count % (self.budget // 10) == 0:
                diversity = np.std(population, axis=0).mean()
                if diversity < self.diversity_threshold or fitness_variance < 0.01:
                    self.population_size = min(self.initial_population_size, int(self.population_size * 1.3))  # Further reduced growth multiplier
                else:
                    self.population_size = max(self.min_population_size, self.population_size // 2)
                indices = np.argsort(fitness)[:self.population_size]
                population = population[indices]
                fitness = fitness[indices]

            self.crossover_rate = 0.3 + 0.4 * (1 - fitness_variance / (fitness_variance + 1))  # Adjusted crossover scaling
            self.mutation_factor = 0.3 + 0.7 * (1 - diversity / (diversity + 1))  # Modified mutation scaling

            elite_size = max(1, self.population_size // 8)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_population = population[elite_indices]

            if diversity < self.diversity_threshold:
                new_individuals = np.random.rand(self.initial_population_size - self.population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
                population = np.vstack((elite_population, new_individuals))
                fitness = np.append(fitness[elite_indices], [func(ind) for ind in new_individuals])
                self.eval_count += len(new_individuals)
            else:
                elite_population = np.vstack((elite_population, elite_population + np.random.randn(elite_size, self.dim) * 0.05))  # Reduced noise scale
                population = np.vstack((elite_population, population))
                fitness = np.append(fitness[elite_indices], fitness)

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]