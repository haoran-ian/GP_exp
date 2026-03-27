import numpy as np

class RefinedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.8  # Increased for DE strategy
        self.crossover_rate = 0.9  # Added for DE strategy
        self.elite_rate = 0.2
        self.local_search_rate = 0.3
        self.adaptive_threshold = 0.1

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_elite(self, population, fitness):
        elite_count = int(self.elite_rate * self.population_size)
        elite_indices = fitness.argsort()[:elite_count]
        return population[elite_indices]

    def _differential_evolution_mutation(self, target_idx, population, bounds):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.mutation_rate * (b - c), bounds.lb, bounds.ub)
        return mutant

    def _crossover(self, target, mutant):
        crossover_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, target)
        return crossover_vector

    def _local_search(self, individual, func, bounds):
        step_size = (bounds.ub - bounds.lb) * 0.05
        local = individual + np.random.uniform(-step_size, step_size, size=self.dim)
        local_fitness = func(local)
        if local_fitness < func(individual):
            return local
        return individual

    def _adaptive_parameter_tuning(self, fitness):
        if np.std(fitness) < self.adaptive_threshold:
            self.mutation_rate = min(1.0, self.mutation_rate + 0.05)
            self.crossover_rate = max(0.1, self.crossover_rate - 0.05)
        else:
            self.mutation_rate = max(0.5, self.mutation_rate - 0.05)
            self.crossover_rate = min(1.0, self.crossover_rate + 0.05)

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(population, func)
        evaluations = self.population_size

        while evaluations < self.budget:
            elite = self._select_elite(population, fitness)
            new_population = elite.copy()
            self._adaptive_parameter_tuning(fitness)
            while len(new_population) < self.population_size:
                if np.random.rand() < self.local_search_rate:
                    parent = elite[np.random.randint(len(elite))]
                    offspring = self._local_search(parent, func, bounds)
                else:
                    target_idx = np.random.randint(self.population_size)
                    target = population[target_idx]
                    mutant = self._differential_evolution_mutation(target_idx, population, bounds)
                    offspring = self._crossover(target, mutant)
                
                if evaluations < self.budget:
                    new_population = np.vstack([new_population, offspring])
                    fitness = np.append(fitness, func(offspring))
                    evaluations += 1

            population = new_population
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]