import numpy as np

class RefinedHierarchicalAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_temp = 100.0
        self.cooling_rate = 0.98
        self.bounds = None
        self.elitism_rate = 0.2
        self.groups = 5
        self.levy_alpha = 1.5

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _adaptive_mutation(self, fitness, f_min=0.1, f_max=0.9):
        norm_fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-6)
        return f_min + (f_max - f_min) * (1 - norm_fitness**2)

    def _levy_flight(self, size, scale):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v)**(1 / beta)
        return scale * step * self.levy_alpha

    def _mutate(self, target_idx, population, fitness, scale):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = self._adaptive_mutation(fitness[target_idx])
        mutant = population[a] + f * (population[b] - population[c]) + self._levy_flight(self.dim, scale)
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _adaptive_crossover(self, evaluations, cr_base=0.5, cr_max=1.0):
        return cr_base + (cr_max - cr_base) * np.sin(np.pi * evaluations / (2 * self.budget))

    def _crossover(self, target, mutant, cr):
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

            current_groups = max(2, int(self.groups * (1 - evaluations / self.budget)))
            scale = 1 + 0.5 * np.random.rand() * np.sin(evaluations / self.budget * np.pi)
            cr = self._adaptive_crossover(evaluations)

            for g in range(current_groups):
                group_indices = np.random.permutation(range(g, self.population_size, current_groups))

                for i in group_indices:
                    if i < elite_size:
                        continue

                    mutant = self._mutate(i, population, fitness, scale)
                    trial = self._crossover(population[i], mutant, cr)
                    new_population[i], new_fitness[i] = self._anneal(trial, population[i], func, temperature)
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

            combined_pop = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            ranks = np.argsort(np.argsort(combined_fitness))
            probabilities = 1 / (ranks + 1)
            probabilities /= probabilities.sum()
            selected_indices = np.random.choice(len(combined_pop), self.population_size, p=probabilities)
            population = combined_pop[selected_indices]
            fitness = combined_fitness[selected_indices]

            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]