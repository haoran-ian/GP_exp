import numpy as np

class EnhancedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.max_population_size = 200
        self.min_population_size = 10
        self.mutation_rate = 0.1
        self.elite_rate = 0.2
        self.local_search_rate = 0.3
        self.convergence_threshold = 0.01

    def _initialize_population(self, bounds, size):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_elite(self, population, fitness):
        elite_count = int(self.elite_rate * len(population))
        elite_indices = fitness.argsort()[:elite_count]
        return population[elite_indices]

    def _mutate(self, individual, bounds, adaptation_factor):
        lb, ub = bounds.lb, bounds.ub
        noise = np.random.normal(0, 1, size=self.dim) * self.mutation_rate * adaptation_factor
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
        population_size = self.initial_population_size
        population = self._initialize_population(bounds, population_size)
        fitness = self._evaluate_population(population, func)
        evaluations = population_size
        last_best_fitness = np.min(fitness)

        while evaluations < self.budget:
            elite = self._select_elite(population, fitness)
            new_population = elite.copy()
            current_best_fitness = np.min(fitness)
            convergence_speed = abs(last_best_fitness - current_best_fitness)
            adaptation_factor = 1 - min(self.convergence_threshold, convergence_speed) / self.convergence_threshold
            population_size = max(self.min_population_size, min(self.max_population_size, int(population_size * (1 + adaptation_factor))))

            while len(new_population) < population_size:
                if np.random.rand() < self.local_search_rate:
                    parent = elite[np.random.randint(len(elite))]
                    offspring = self._local_search(parent, func, bounds)
                else:
                    parent1, parent2 = elite[np.random.choice(len(elite), 2, replace=False)]
                    offspring = self._crossover(parent1, parent2)
                    offspring = self._mutate(offspring, bounds, adaptation_factor)
                
                if evaluations < self.budget:
                    new_population = np.vstack([new_population, offspring])
                    fitness = np.append(fitness, func(offspring))
                    evaluations += 1

            population = new_population
            fitness = self._evaluate_population(population, func)
            last_best_fitness = current_best_fitness

        best_index = np.argmin(fitness)
        return population[best_index]