import numpy as np

class EnhancedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.base_mutation_rate = 0.1
        self.elite_rate = 0.2
        self.local_search_rate = 0.3
        self.convergence_threshold = 0.01

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_elite(self, population, fitness):
        elite_count = int(self.elite_rate * self.population_size)
        elite_indices = fitness.argsort()[:elite_count]
        return population[elite_indices]

    def _adaptive_mutation(self, individual, bounds, convergence_speed):
        lb, ub = bounds.lb, bounds.ub
        adaptive_mutation_rate = self.base_mutation_rate * (1 - convergence_speed)
        noise = np.random.normal(0, 1, size=self.dim) * adaptive_mutation_rate
        mutated = np.clip(individual + noise, lb, ub)
        return mutated

    def _crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        return alpha * parent1 + (1 - alpha) * parent2

    def _dynamic_local_search(self, individual, func, bounds, convergence_speed):
        intensity = (bounds.ub - bounds.lb) * (0.05 + 0.05 * convergence_speed)
        local = individual + np.random.uniform(-intensity, intensity, size=self.dim)
        local_fitness = func(local)
        if local_fitness < func(individual):
            return local
        return individual

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(population, func)
        evaluations = self.population_size
        previous_best = np.min(fitness)

        while evaluations < self.budget:
            elite = self._select_elite(population, fitness)
            new_population = elite.copy()
            current_best = np.min(fitness)
            convergence_speed = abs(previous_best - current_best) / max(previous_best, 1e-10)
            previous_best = current_best

            while len(new_population) < self.population_size:
                if np.random.rand() < self.local_search_rate:
                    parent = elite[np.random.randint(len(elite))]
                    offspring = self._dynamic_local_search(parent, func, bounds, convergence_speed)
                else:
                    parent1, parent2 = elite[np.random.choice(len(elite), 2, replace=False)]
                    offspring = self._crossover(parent1, parent2)
                    offspring = self._adaptive_mutation(offspring, bounds, convergence_speed)
                
                if evaluations < self.budget:
                    new_population = np.vstack([new_population, offspring])
                    fitness = np.append(fitness, func(offspring))
                    evaluations += 1

            population = new_population
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]