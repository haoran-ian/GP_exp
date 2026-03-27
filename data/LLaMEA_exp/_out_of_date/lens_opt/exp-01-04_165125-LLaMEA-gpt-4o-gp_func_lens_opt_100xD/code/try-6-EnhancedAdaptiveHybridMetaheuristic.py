import numpy as np

class EnhancedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.initial_mutation_rate = 0.1
        self.initial_elite_rate = 0.2
        self.local_search_rate = 0.3

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_elite(self, population, fitness, elite_rate):
        elite_count = int(elite_rate * self.population_size)
        elite_indices = fitness.argsort()[:elite_count]
        return population[elite_indices]

    def _mutate(self, individual, bounds, mutation_rate):
        lb, ub = bounds.lb, bounds.ub
        noise = np.random.normal(0, 1, size=self.dim) * mutation_rate
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

    def _adapt_parameters(self, evaluations, best_fitness, prev_best_fitness):
        progress = (prev_best_fitness - best_fitness) / prev_best_fitness if prev_best_fitness != 0 else 0
        mutation_rate = self.initial_mutation_rate * (1 - progress)
        elite_rate = self.initial_elite_rate * (1 + progress)
        return mutation_rate, elite_rate

    def _diversity_preservation(self, population, bounds):
        lb, ub = bounds.lb, bounds.ub
        diversity_threshold = 0.1 * (ub - lb)
        unique_population = np.unique(population, axis=0)
        if len(unique_population) < self.population_size:
            new_individuals = np.random.uniform(lb, ub, (self.population_size - len(unique_population), self.dim))
            population = np.vstack([unique_population, new_individuals])
        return population

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(population, func)
        evaluations = self.population_size
        prev_best_fitness = np.inf

        while evaluations < self.budget:
            best_fitness = np.min(fitness)
            mutation_rate, elite_rate = self._adapt_parameters(evaluations, best_fitness, prev_best_fitness)
            elite = self._select_elite(population, fitness, elite_rate)
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                if np.random.rand() < self.local_search_rate:
                    parent = elite[np.random.randint(len(elite))]
                    offspring = self._local_search(parent, func, bounds)
                else:
                    parent1, parent2 = elite[np.random.choice(len(elite), 2, replace=False)]
                    offspring = self._crossover(parent1, parent2)
                    offspring = self._mutate(offspring, bounds, mutation_rate)
                
                if evaluations < self.budget:
                    new_population = np.vstack([new_population, offspring])
                    fitness = np.append(fitness, func(offspring))
                    evaluations += 1

            population = self._diversity_preservation(new_population, bounds)
            fitness = self._evaluate_population(population, func)
            prev_best_fitness = best_fitness

        best_index = np.argmin(fitness)
        return population[best_index]