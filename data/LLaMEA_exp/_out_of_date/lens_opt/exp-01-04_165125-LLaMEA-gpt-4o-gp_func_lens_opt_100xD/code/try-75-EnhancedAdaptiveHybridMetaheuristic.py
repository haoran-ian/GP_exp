import numpy as np

class EnhancedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.elite_rate = 0.2
        self.local_search_rate = 0.3
        self.differential_weight = 0.5
        self.crossover_rate = 0.7
        self.adaptation_frequency = 10

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

    def _differential_evolution(self, population, bounds):
        lb, ub = bounds.lb, bounds.ub
        mutant_population = population.copy()
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[indices]
            mutant_vector = x1 + self.differential_weight * (x2 - x3)
            crossover_prob = np.random.rand(self.dim) < self.crossover_rate
            trial_vector = np.where(crossover_prob, mutant_vector, population[i])
            mutant_population[i] = np.clip(trial_vector, lb, ub)
        return mutant_population

    def _local_search(self, individual, func, bounds):
        step_size = (bounds.ub - bounds.lb) * 0.05
        local = individual + np.random.uniform(-step_size, step_size, size=self.dim)
        local_fitness = func(local)
        if local_fitness < func(individual):
            return local
        return individual

    def _adaptive_parameter_tuning(self, evaluations):
        if evaluations % self.adaptation_frequency == 0:
            self.mutation_rate = np.random.uniform(0.05, 0.2)
            self.differential_weight = np.random.uniform(0.4, 0.9)
            self.crossover_rate = np.random.uniform(0.5, 0.9)

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(population, func)
        evaluations = self.population_size

        while evaluations < self.budget:
            self._adaptive_parameter_tuning(evaluations)
            elite = self._select_elite(population, fitness)
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                if np.random.rand() < self.local_search_rate:
                    parent = elite[np.random.randint(len(elite))]
                    offspring = self._local_search(parent, func, bounds)
                else:
                    if np.random.rand() < 0.5:
                        offspring = self._differential_evolution(elite, bounds)
                        offspring = offspring[np.random.randint(len(elite))]
                    else:
                        parent1, parent2 = elite[np.random.choice(len(elite), 2, replace=False)]
                        offspring = self._crossover(parent1, parent2)
                        offspring = self._mutate(offspring, bounds)
                
                if evaluations < self.budget:
                    new_population = np.vstack([new_population, offspring])
                    fitness = np.append(fitness, func(offspring))
                    evaluations += 1

            population = new_population
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]