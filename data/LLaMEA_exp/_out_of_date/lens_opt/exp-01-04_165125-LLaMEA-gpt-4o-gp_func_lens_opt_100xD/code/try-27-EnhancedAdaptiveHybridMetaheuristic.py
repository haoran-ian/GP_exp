import numpy as np

class EnhancedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.base_mutation_rate = 0.1
        self.elite_rate = 0.2
        self.local_search_rate = 0.3

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_elite(self, population, fitness):
        elite_count = int(self.elite_rate * self.population_size)
        elite_indices = fitness.argsort()[:elite_count]
        return population[elite_indices]

    def _mutate(self, individual, bounds, fitness, best_fitness):
        lb, ub = bounds.lb, bounds.ub
        dynamic_mutation_rate = self.base_mutation_rate * (1 - fitness / best_fitness)
        noise = np.random.normal(0, 1, size=self.dim) * dynamic_mutation_rate
        mutated = np.clip(individual + noise, lb, ub)
        return mutated

    def _crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        return alpha * parent1 + (1 - alpha) * parent2

    def _local_search(self, individual, func, bounds, fitness, best_fitness):
        adj_factor = (fitness / best_fitness) if fitness < best_fitness else 1
        step_size = (bounds.ub - bounds.lb) * 0.05 * adj_factor
        local = individual + np.random.uniform(-step_size, step_size, size=self.dim)
        local_fitness = func(local)
        if local_fitness < func(individual):
            return local
        return individual

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(population, func)
        evaluations = self.population_size

        while evaluations < self.budget:
            elite = self._select_elite(population, fitness)
            new_population = elite.copy()
            best_fitness = min(fitness)
            while len(new_population) < self.population_size:
                if np.random.rand() < self.local_search_rate:
                    parent = elite[np.random.randint(len(elite))]
                    offspring = self._local_search(parent, func, bounds, func(parent), best_fitness)
                else:
                    parent1, parent2 = elite[np.random.choice(len(elite), 2, replace=False)]
                    offspring = self._crossover(parent1, parent2)
                    offspring = self._mutate(offspring, bounds, func(offspring), best_fitness)
                
                if evaluations < self.budget:
                    new_population = np.vstack([new_population, offspring])
                    fitness = np.append(fitness, func(offspring))
                    evaluations += 1

            population = new_population
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]