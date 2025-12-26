import numpy as np

class HierarchicalAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_temp = 100.0
        self.cooling_rate = 0.95
        self.bounds = None
        self.elitism_rate = 0.2
        self.groups = 5  # Hierarchical grouping

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _adaptive_mutation(self, fitness, f_min=0.3, f_max=0.8):
        norm_fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-6)
        return f_min + (f_max - f_min) * (1 - norm_fitness)

    def _mutate(self, target_idx, population, fitness):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = self._adaptive_mutation(fitness[target_idx])
        mutant = population[a] + f * (population[b] - population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _crossover(self, target, mutant, cr=0.7):  # Modified crossover probability
        mask = np.random.rand(self.dim) < cr
        return np.where(mask, mutant, target)

    def _acceptance_probability(self, current, candidate, t):
        return 1.0 if candidate < current else np.exp((current - candidate) / t)

    def _anneal(self, candidate, current, func, temperature):
        candidate_fitness = func(candidate)
        if self._acceptance_probability(func(current), candidate_fitness, temperature) > np.random.rand():
            return candidate, candidate_fitness
        return current, func(current)

    def __call__(self, func):
        self.bounds = func.bounds
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        temperature = self.initial_temp
        elite_size = int(self.elitism_rate * self.population_size)

        while evaluations < self.budget:
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            # Group individuals for hierarchical processing
            for g in range(self.groups):
                group_indices = range(g, self.population_size, self.groups)

                for i in group_indices:
                    if i < elite_size:
                        continue  # Preserve elite

                    mutant = self._mutate(i, population, fitness)
                    trial = self._crossover(population[i], mutant)
                    new_population[i], new_fitness[i] = self._anneal(trial, population[i], func, temperature)
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

            combined_pop = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_pop[best_indices]
            fitness = combined_fitness[best_indices]

            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]