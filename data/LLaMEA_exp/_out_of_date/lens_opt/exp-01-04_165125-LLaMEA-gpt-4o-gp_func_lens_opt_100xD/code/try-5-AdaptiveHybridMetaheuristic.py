import numpy as np

class AdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
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

    def _adapt_mutation_rate(self, evaluations, max_evaluations):
        return self.mutation_rate * (1 - evaluations / max_evaluations)

    def _mutate(self, individual, bounds, evaluations, max_evaluations):
        lb, ub = bounds.lb, bounds.ub
        adaptive_mutation_rate = self._adapt_mutation_rate(evaluations, max_evaluations)
        noise = np.random.normal(0, 1, size=self.dim) * adaptive_mutation_rate
        mutated = np.clip(individual + noise, lb, ub)
        return mutated

    def _dynamic_crossover(self, parent1, parent2, evaluations, max_evaluations):
        alpha = np.random.rand(self.dim) * (1 - evaluations / max_evaluations)
        return alpha * parent1 + (1 - alpha) * parent2

    def _local_search(self, individual, func, bounds):
        step_size = (bounds.ub - bounds.lb) * 0.05
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
            while len(new_population) < self.population_size:
                if np.random.rand() < self.local_search_rate:
                    parent = elite[np.random.randint(len(elite))]
                    offspring = self._local_search(parent, func, bounds)
                else:
                    parent1, parent2 = elite[np.random.choice(len(elite), 2, replace=False)]
                    offspring = self._dynamic_crossover(parent1, parent2, evaluations, self.budget)
                    offspring = self._mutate(offspring, bounds, evaluations, self.budget)
                
                if evaluations < self.budget:
                    new_population = np.vstack([new_population, offspring])
                    fitness = np.append(fitness, func(offspring))
                    evaluations += 1

            population = new_population
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]