import numpy as np

class HierarchicalAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_temp = 100.0
        self.cooling_rate = 0.9  # Faster cooling
        self.bounds = None
        self.elitism_rate = 0.2
        self.groups = 5
        self.dynamic_grouping = True

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _adaptive_mutation(self, fitness, diversity, f_min=0.3, f_max=0.8):
        norm_fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-6)
        adaptive_scale = (1 - diversity) * (f_max - f_min)
        return f_min + adaptive_scale * (1 - norm_fitness)

    def _mutate(self, target_idx, population, fitness, diversity):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = self._adaptive_mutation(fitness[target_idx], diversity)
        mutant = population[a] + f * (population[b] - population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _crossover(self, target, mutant, cr=0.7):  # Increased crossover rate
        mask = np.random.rand(self.dim) < cr
        return np.where(mask, mutant, target)

    def _acceptance_probability(self, current, candidate, t):
        return 1.0 if candidate < current else np.exp((current - candidate) / (t + 1e-6))

    def _anneal(self, candidate, current, func, temperature):
        candidate_fitness = func(candidate)
        if self._acceptance_probability(func(current), candidate_fitness, temperature) > np.random.rand():
            return candidate, candidate_fitness
        return current, func(current)

    def _calculate_diversity(self, population):
        pairwise_distances = np.linalg.norm(population[:, np.newaxis] - population, axis=-1)
        return np.mean(pairwise_distances) / (np.abs(self.bounds.ub - self.bounds.lb) + 1e-6)

    def __call__(self, func):
        self.bounds = func.bounds
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        temperature = self.initial_temp
        elite_size = int(self.elitism_rate * self.population_size)

        while evaluations < self.budget:
            diversity = self._calculate_diversity(population)
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            effective_groups = max(1, int(self.groups * (1 - diversity)))
            for g in range(effective_groups):
                group_indices = range(g, self.population_size, effective_groups)

                for i in group_indices:
                    if i < elite_size:
                        continue

                    mutant = self._mutate(i, population, fitness, diversity)
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