import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
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

            # Adaptive mutation and crossover based on success history
            mutation_factor = np.clip(np.random.normal(self.mutation_factor, 0.1), 0.5, 1.0)
            crossover_prob = np.clip(np.random.normal(self.crossover_prob, 0.1), 0.5, 1.0)

            offspring = self._differential_evolution(self.population, fitness, mutation_factor, crossover_prob, bounds)
            
            if elite_count > 0:
                offspring[:elite_count] = elite_population

            # Introduce adaptive local search around the best solution found
            for i in range(elite_count):
                gradient_step = 0.01 * (self.best_solution - offspring[i])
                offspring[i] = np.clip(offspring[i] + gradient_step, bounds[0], bounds[1])

            self.population = offspring

        return self.best_solution

    def _differential_evolution(self, population, fitness, mutation_factor, crossover_prob, bounds):
        offspring = np.empty_like(population)
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = population[indices]
            mutant_vector = np.clip(a + mutation_factor * (b - c), bounds[0], bounds[1])
            cross_points = np.random.rand(self.dim) < crossover_prob
            offspring[i] = np.where(cross_points, mutant_vector, population[i])
        return offspring