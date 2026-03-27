import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.8
        self.crossover_rate = 0.9
        self.local_search_rate = 0.3
        self.elite_rate = 0.2

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_elite(self, population, fitness):
        elite_count = int(self.elite_rate * self.population_size)
        elite_indices = fitness.argsort()[:elite_count]
        return population[elite_indices]

    def _mutate(self, target_idx, population, bounds):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        diff = population[a] + self.mutation_rate * (population[b] - population[c])
        return np.clip(diff, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def _adaptive_local_search(self, individual, func, bounds):
        best_individual = individual
        best_fitness = func(individual)
        step_factor = 0.05
        while step_factor >= 0.001:
            step_size = (bounds.ub - bounds.lb) * step_factor
            local = best_individual + np.random.uniform(-step_size, step_size, size=self.dim)
            local_fitness = func(local)
            if local_fitness < best_fitness:
                best_individual, best_fitness = local, local_fitness
            step_factor *= 0.5
        return best_individual

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(population, func)
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                mutant = self._mutate(i, population, bounds)
                offspring = self._crossover(population[i], mutant)
                
                if np.random.rand() < self.local_search_rate:
                    offspring = self._adaptive_local_search(offspring, func, bounds)
                
                new_population.append(offspring)
                if evaluations < self.budget:
                    fitness = np.append(fitness, func(offspring))
                    evaluations += 1

            population = np.array(new_population)
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]