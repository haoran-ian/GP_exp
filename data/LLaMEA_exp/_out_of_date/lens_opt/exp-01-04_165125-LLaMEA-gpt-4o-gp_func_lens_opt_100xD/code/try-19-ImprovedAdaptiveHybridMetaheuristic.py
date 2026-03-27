import numpy as np

class ImprovedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.elite_rate = 0.2
        self.local_search_rate = 0.3
        self.adaptive_factor = 0.95  # New parameter for adaptive tuning

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

    def _enhanced_local_search(self, individual, func, bounds):
        step_scale = np.linspace(0.05, 0.01, 5)  # More steps with decreasing scale
        current_solution = individual
        current_fitness = func(current_solution)
        for step_size in step_scale:
            local = current_solution + np.random.uniform(-step_size, step_size, size=self.dim) * (bounds.ub - bounds.lb)
            local_fitness = func(local)
            if local_fitness < current_fitness:
                current_solution, current_fitness = local, local_fitness
        return current_solution

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
                    offspring = self._enhanced_local_search(parent, func, bounds)
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

            # Adapt parameters based on current search performance
            self.mutation_rate *= self.adaptive_factor
            self.local_search_rate *= self.adaptive_factor

        best_index = np.argmin(fitness)
        return population[best_index]