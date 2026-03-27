import numpy as np

class EnhancedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_rate = 0.2
        self.local_search_rate = 0.3
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_elite(self, population, fitness):
        elite_count = int(self.elite_rate * self.population_size)
        elite_indices = fitness.argsort()[:elite_count]
        return population[elite_indices]

    def _mutate(self, population, elite, bounds):
        lb, ub = bounds.lb, bounds.ub
        mutated = []
        for target in population:
            a, b, c = elite[np.random.choice(len(elite), 3, replace=False)]
            donor_vector = a + self.mutation_factor * (b - c)
            donor_vector = np.clip(donor_vector, lb, ub)
            mutated.append(donor_vector)
        return mutated

    def _crossover(self, target, donor):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        return np.where(crossover_mask, donor, target)

    def _local_search(self, individual, func, bounds):
        step_size = (bounds.ub - bounds.lb) * 0.05
        for _ in range(5):  # Small local search steps
            step = np.random.uniform(-step_size, step_size, size=self.dim)
            candidate = np.clip(individual + step, bounds.lb, bounds.ub)
            if func(candidate) < func(individual):
                return candidate
        return individual

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(population, func)
        evaluations = self.population_size

        while evaluations < self.budget:
            elite = self._select_elite(population, fitness)
            mutants = self._mutate(population, elite, bounds)
            new_population = []

            for target, donor in zip(population, mutants):
                offspring = self._crossover(target, donor)
                if np.random.rand() < self.local_search_rate:
                    offspring = self._local_search(offspring, func, bounds)

                if evaluations < self.budget:
                    new_population.append(offspring)
                    evaluations += 1

            population = np.array(new_population)
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]