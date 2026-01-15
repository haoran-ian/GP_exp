import numpy as np

class EnhancedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
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

    def _dynamic_mutation(self, individual, bounds, progress):
        lb, ub = bounds.lb, bounds.ub
        mutation_rate = 0.1 * (1 - progress)  # Decrease mutation rate as progress increases
        noise = np.random.normal(0, 1, size=self.dim) * mutation_rate
        mutated = np.clip(individual + noise, lb, ub)
        return mutated

    def _crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        return alpha * parent1 + (1 - alpha) * parent2

    def _adaptive_local_search(self, individual, func, bounds, best_fitness, current_fitness):
        improvement = (best_fitness - current_fitness) / abs(best_fitness) if best_fitness != 0 else 0
        step_size = (bounds.ub - bounds.lb) * (0.1 * improvement)
        local = individual + np.random.uniform(-step_size, step_size, size=self.dim)
        local_fitness = func(local)
        if local_fitness < current_fitness:
            return local
        return individual

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(population, func)
        evaluations = self.population_size
        best_fitness = np.min(fitness)

        while evaluations < self.budget:
            elite = self._select_elite(population, fitness)
            new_population = elite.copy()
            progress = evaluations / self.budget
            while len(new_population) < self.population_size and evaluations < self.budget:
                if np.random.rand() < self.local_search_rate:
                    parent_idx = np.random.randint(len(elite))
                    parent = elite[parent_idx]
                    offspring = self._adaptive_local_search(parent, func, bounds, best_fitness, fitness[parent_idx])
                else:
                    parent1, parent2 = elite[np.random.choice(len(elite), 2, replace=False)]
                    offspring = self._crossover(parent1, parent2)
                    offspring = self._dynamic_mutation(offspring, bounds, progress)
                new_population = np.vstack([new_population, offspring])
                offspring_fitness = func(offspring)
                fitness = np.append(fitness, offspring_fitness)
                best_fitness = min(best_fitness, offspring_fitness)
                evaluations += 1

            population = new_population
            fitness = self._evaluate_population(population, func)

        best_index = np.argmin(fitness)
        return population[best_index]