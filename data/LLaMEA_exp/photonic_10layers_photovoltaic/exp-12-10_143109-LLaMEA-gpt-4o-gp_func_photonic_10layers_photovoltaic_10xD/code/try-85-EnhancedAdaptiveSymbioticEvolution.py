import numpy as np

class EnhancedAdaptiveSymbioticEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.mutation_rate = 0.5
        self.crossover_rate = 0.7
        self.elitism_rate = 0.1
        self.population = None
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        self.population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        evaluations = 0

        while evaluations < self.budget:
            fitness = np.apply_along_axis(func, 1, self.population)
            evaluations += self.population_size

            elite_count = max(1, int(self.elitism_rate * self.population_size))
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_population = self.population[elite_indices]

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.best_fitness = fitness[min_idx]
                self.best_solution = self.population[min_idx].copy()

            if evaluations > self.budget // 10:
                avg_fitness = np.mean(fitness)
                fitness_std = np.std(fitness)
                diversity = np.std(self.population, axis=0).mean()

                self.mutation_rate = 0.1 + 0.9 * (1.0 - fitness_std / (np.abs(avg_fitness) + 1e-8))
                self.crossover_rate = 0.5 + 0.5 * (diversity / (np.max(np.std(self.population, axis=0)) + 1e-8))

                if fitness_std < 0.1 and self.population_size > 5:
                    self.population_size = max(5, self.population_size - 1)
                elif fitness_std > 0.5 and evaluations + self.population_size <= self.budget:
                    self.population_size = min(100, self.population_size + 1)

                fitness_gradient = np.gradient(fitness)
                improvement_ratio = (fitness[min_idx] - self.best_fitness) / (np.abs(avg_fitness) + 1e-8)
                if improvement_ratio < 0:
                    self.mutation_rate = min(1.0, self.mutation_rate * 1.1)
                    self.crossover_rate = max(0.1, self.crossover_rate * 0.9)

            parents = self._select_parents(fitness)
            offspring = self._differential_evolution(parents, bounds)
            offspring[:elite_count] = elite_population

            adaptive_learning_rate = 0.01 * (1.0 / (1.0 + np.exp(-5 * improvement_ratio)))
            offspring[min_idx] -= adaptive_learning_rate * fitness_gradient[min_idx]

            self.population = offspring

        return self.best_solution

    def _select_parents(self, fitness):
        tournament_size = 3
        indices = np.random.randint(0, len(self.population), (self.population_size, tournament_size))
        tournament_fitness = fitness[indices]
        winners = indices[np.arange(self.population_size), np.argmin(tournament_fitness, axis=1)]
        return self.population[winners]

    def _differential_evolution(self, parents, bounds):
        offspring = np.empty_like(parents)
        for i in range(self.population_size):
            idxs = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[idxs]
            mutant = np.clip(a + self.mutation_rate * (b - c), bounds[0], bounds[1])
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            offspring[i] = np.where(cross_points, mutant, parents[i])
        return offspring