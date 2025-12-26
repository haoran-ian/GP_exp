import numpy as np

class RefinedHierarchicalAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_temp = 100.0
        self.cooling_rate = 0.97
        self.bounds = None
        self.elitism_rate = 0.25
        self.groups = 5
        self.levy_alpha = 1.5
        self.local_search_probability = 0.3

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _adaptive_mutation(self, fitness, f_min=0.3, f_max=0.8):
        norm_fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-6)
        return f_min + (f_max - f_min) * (1 - norm_fitness)

    def _levy_flight(self, size, scale):
        return np.random.standard_cauchy(size) * self.levy_alpha * scale

    def _mutate(self, target_idx, population, fitness, scale):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = self._adaptive_mutation(fitness[target_idx])
        mutant = population[a] + f * (population[b] - population[c]) + self._levy_flight(self.dim, scale)
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _adaptive_crossover(self, evaluations, cr_base=0.65, cr_max=0.9):
        return cr_base + (cr_max - cr_base) * (evaluations / self.budget)

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

    def _stochastic_local_search(self, individual, func):
        perturbation = np.random.normal(scale=0.1, size=self.dim)
        candidate = np.clip(individual + perturbation, self.bounds.lb, self.bounds.ub)
        candidate_fitness = func(candidate)
        return candidate, candidate_fitness

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
            scale = 1 + 0.5 * np.sin(evaluations / self.budget * np.pi)
            cr = self._adaptive_crossover(evaluations)

            for g in range(current_groups):
                group_indices = range(g, self.population_size, current_groups)

                for i in group_indices:
                    if i < elite_size:
                        continue

                    if np.random.rand() < self.local_search_probability:
                        trial, trial_fitness = self._stochastic_local_search(population[i], func)
                    else:
                        mutant = self._mutate(i, population, fitness, scale)
                        trial = self._crossover(population[i], mutant, cr)
                        trial, trial_fitness = self._anneal(trial, population[i], func, temperature)

                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
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