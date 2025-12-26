import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_temp = 100.0
        self.cooling_rate = 0.95
        self.bounds = None

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _mutate(self, target_idx, population, f_min=0.5, f_max=0.9):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = np.random.uniform(f_min, f_max)
        mutant = population[a] + f * (population[b] - population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _crossover(self, target, mutant, cr=0.5):
        mask = np.random.rand(self.dim) < cr
        return np.where(mask, mutant, target)

    def _acceptance_probability(self, current, candidate, t):
        if candidate < current:
            return 1.0
        else:
            return np.exp((current - candidate) / t)

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

        while evaluations < self.budget:
            for i in range(self.population_size):
                f_min_adaptive = 0.4 + 0.5 * (1 - evaluations / self.budget)
                f_max_adaptive = 0.8 + 0.2 * (1 - evaluations / self.budget)
                mutant = self._mutate(i, population, f_min=f_min_adaptive, f_max=f_max_adaptive)
                cr_adaptive = 0.3 + 0.7 * (1 - fitness[i] / np.max(fitness))
                trial = self._crossover(population[i], mutant, cr=cr_adaptive)
                population[i], fitness[i] = self._anneal(trial, population[i], func, temperature)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            
            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]