import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
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
        self.diversity_threshold = 0.1
        self.reinit_rate = 0.05  # New parameter for periodic reinitialization

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.rand(self.population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size
        
        p_best_rate = 0.2

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                oscillating_factor = np.sin(2 * np.pi * self.eval_count / self.budget)
                self_adaptive_mutation = self.alpha * np.random.randn(self.dim)
                
                p_best = population[np.argsort(fitness)[:max(1, int(p_best_rate * self.population_size))]][0]
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3) * oscillating_factor + self_adaptive_mutation + 0.5 * (p_best - x1), bounds[0], bounds[1])

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

            if self.eval_count % (self.budget // 10) == 0 and self.population_size > self.min_population_size:
                self.population_size = max(self.min_population_size, self.population_size // 2)
                indices = np.argsort(fitness)[:self.population_size]
                population = population[indices]
                fitness = fitness[indices]

            # Self-adaptive rates
            self.crossover_rate = 0.3 + 0.4 * np.sin(2 * np.pi * self.eval_count / self.budget)
            self.mutation_factor = 0.5 + 0.3 * np.cos(2 * np.pi * (self.eval_count/self.budget) * (fitness.mean() / (fitness.min() + 1e-8)))

            elite_size = max(1, self.population_size // 10)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_population = population[elite_indices]
            population_diversity = np.std(population, axis=0).mean()

            if population_diversity < self.diversity_threshold:
                new_individuals = np.random.rand(self.initial_population_size - self.population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
                population = np.vstack((elite_population, new_individuals))
                fitness = np.append(fitness[elite_indices], [func(ind) for ind in new_individuals])
                self.eval_count += len(new_individuals)
            else:
                population = np.vstack((elite_population, population))
                fitness = np.append(fitness[elite_indices], fitness)

            # Periodic reinitialization for diversity
            if self.eval_count % int(self.budget * self.reinit_rate) == 0:
                num_to_reinit = max(1, int(self.reinit_rate * self.population_size))
                reinit_indices = np.random.choice(self.population_size, num_to_reinit, replace=False)
                population[reinit_indices] = np.random.rand(num_to_reinit, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
                fitness[reinit_indices] = [func(ind) for ind in population[reinit_indices]]
                self.eval_count += num_to_reinit

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]