import numpy as np

class EnhancedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_rate = 0.2
        self.local_search_rate = 0.3
        self.differential_weight = 0.8
        self.crossover_probability = 0.9

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_elite(self, population, fitness):
        elite_count = int(self.elite_rate * self.population_size)
        elite_indices = fitness.argsort()[:elite_count]
        return population[elite_indices]

    def _differential_evolution(self, target, donor, bounds):
        trial = np.where(np.random.rand(self.dim) < self.crossover_probability, donor, target)
        return np.clip(trial, bounds.lb, bounds.ub)

    def _local_search(self, individual, func, bounds):
        step_size = (bounds.ub - bounds.lb) * 0.05
        for _ in range(5):  # Increase local search steps
            local = individual + np.random.uniform(-step_size, step_size, size=self.dim)
            if func(local) < func(individual):
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
                    indices = np.random.choice(len(elite), 3, replace=False)
                    a, b, c = elite[indices]
                    donor = a + self.differential_weight * (b - c)
                    target = elite[np.random.randint(len(elite))]
                    offspring = self._differential_evolution(target, donor, bounds)

                if evaluations < self.budget:
                    new_population = np.vstack([new_population, offspring])
                    fitness = np.append(fitness, func(offspring))
                    evaluations += 1

            population = new_population
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]