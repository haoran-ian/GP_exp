import numpy as np

class RefinedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.mutation_rate = 0.1
        self.elite_rate = 0.2
        self.local_search_rate = 0.3
        self.adaptive_factor = 0.05

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_elite(self, population, fitness):
        elite_count = int(self.elite_rate * len(population))
        elite_indices = fitness.argsort()[:elite_count]
        return population[elite_indices]

    def _mutate(self, individual, bounds, success_rate):
        lb, ub = bounds.lb, bounds.ub
        adapt_mutation_rate = self.mutation_rate * (1 + self.adaptive_factor * (1 - success_rate))
        noise = np.random.normal(0, 1, size=self.dim) * adapt_mutation_rate
        mutated = np.clip(individual + noise, lb, ub)
        return mutated

    def _crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
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
        evaluations = len(population)

        success_count = 0
        while evaluations < self.budget:
            elite = self._select_elite(population, fitness)
            new_population = elite.copy()
            success_rate = success_count / len(population)
            success_count = 0  # Reset for the new iteration

            while len(new_population) < len(population):
                if np.random.rand() < self.local_search_rate:
                    parent = elite[np.random.randint(len(elite))]
                    offspring = self._local_search(parent, func, bounds)
                else:
                    parent1, parent2 = elite[np.random.choice(len(elite), 2, replace=False)]
                    offspring = self._crossover(parent1, parent2)
                    offspring = self._mutate(offspring, bounds, success_rate)
                
                if evaluations < self.budget:
                    new_population = np.vstack([new_population, offspring])
                    offspring_fitness = func(offspring)
                    fitness = np.append(fitness, offspring_fitness)
                    evaluations += 1
                    if offspring_fitness < np.min(fitness):
                        success_count += 1

            # Dynamically adjust population size
            if success_rate > 0.5:
                self.initial_population_size = min(self.initial_population_size + 1, 100)
            else:
                self.initial_population_size = max(self.initial_population_size - 1, 20)

            population = new_population
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]