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
        self.historical_mutation_strength = []

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
                self.crossover_rate = max(0.3, self.crossover_rate - 0.1 * (fitness_std / (np.abs(avg_fitness) + 1e-8)))

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
            offspring = self._crossover(parents, bounds)

            # Historical mutation strength adjustment
            avg_improvement = self._calculate_average_improvement(fitness)
            mutation_strength_scaler = self._adjust_mutation_strength(avg_improvement)
            self.historical_mutation_strength.append(mutation_strength_scaler)
            if len(self.historical_mutation_strength) > 10:
                self.historical_mutation_strength.pop(0)

            offspring = self._mutate(offspring, bounds, mutation_strength_scaler)

            if elite_count > 0:
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

    def _crossover(self, parents, bounds):
        offspring = np.empty_like(parents)
        for i in range(0, self.population_size, 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.dim)
                offspring[i][:crossover_point] = parents[i][:crossover_point]
                offspring[i][crossover_point:] = parents[i+1][crossover_point:]
                offspring[i+1][:crossover_point] = parents[i+1][:crossover_point]
                offspring[i+1][crossover_point:] = parents[i][crossover_point:]
            else:
                offspring[i], offspring[i+1] = parents[i], parents[i+1]
        return np.clip(offspring, bounds[0], bounds[1])

    def _mutate(self, offspring, bounds, mutation_strength_scaler):
        mutation_strength = (bounds[1] - bounds[0]) * self.mutation_rate
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate * mutation_strength_scaler:
                offspring[i] += np.random.normal(0, mutation_strength * mutation_strength_scaler, self.dim)
        return np.clip(offspring, bounds[0], bounds[1])

    def _calculate_average_improvement(self, fitness):
        prev_avg_fitness = np.inf if not self.historical_mutation_strength else np.mean(self.historical_mutation_strength)
        current_avg_fitness = np.mean(fitness)
        improvement = prev_avg_fitness - current_avg_fitness
        return improvement

    def _adjust_mutation_strength(self, improvement):
        if improvement > 0:
            return min(2.0, self.mutation_rate * (1 + improvement))
        else:
            return max(0.1, self.mutation_rate * (1 + improvement))