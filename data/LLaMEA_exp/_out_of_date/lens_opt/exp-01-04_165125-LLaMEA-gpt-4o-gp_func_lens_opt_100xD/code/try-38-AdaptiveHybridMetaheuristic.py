import numpy as np

class AdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.elite_rate = 0.2
        self.local_search_rate = 0.3
        self.adapt_threshold = 0.1  # New parameter for adaptive adjustment

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_elite(self, population, fitness):
        elite_count = int(self.elite_rate * self.population_size)
        elite_indices = fitness.argsort()[:elite_count]
        return population[elite_indices]

    def _mutate(self, individual, bounds):
        lb, ub = bounds.lb, bounds.ub
        noise = np.random.normal(0, 1, size=self.dim) * self.mutation_rate
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

    def _adjust_parameters(self, previous_best, current_best):
        if abs(previous_best - current_best) < self.adapt_threshold:
            self.mutation_rate *= 1.1
            self.local_search_rate *= 0.9
        else:
            self.mutation_rate *= 0.9
            self.local_search_rate *= 1.1

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(population, func)
        evaluations = self.population_size
        previous_best = np.min(fitness)

        while evaluations < self.budget:
            elite = self._select_elite(population, fitness)
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                if np.random.rand() < self.local_search_rate:
                    parent = elite[np.random.randint(len(elite))]
                    offspring = self._local_search(parent, func, bounds)
                else:
                    parent1, parent2 = elite[np.random.choice(len(elite), 2, replace=False)]
                    offspring = self._crossover(parent1, parent2)
                    offspring = self._mutate(offspring, bounds)
                
                if evaluations < self.budget:
                    new_population = np.vstack([new_population, offspring])
                    fitness = np.append(fitness, func(offspring))
                    evaluations += 1

            current_best = np.min(fitness)
            self._adjust_parameters(previous_best, current_best)
            previous_best = current_best
            population = new_population
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]