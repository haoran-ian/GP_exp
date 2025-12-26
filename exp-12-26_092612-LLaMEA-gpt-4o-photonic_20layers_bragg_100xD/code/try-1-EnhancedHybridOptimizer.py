import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.f = 0.8
        self.cr = 0.7
        self.bounds = None

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _mutate(self, target_idx, population):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.f * (population[b] - population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.cr
        return np.where(mask, mutant, target)

    def _acceptance_probability(self, current, candidate):
        if candidate < current:
            return 1.0
        else:
            return np.exp((current - candidate) / self.temperature)

    def _anneal(self, candidate, current, func):
        candidate_fitness = func(candidate)
        if self._acceptance_probability(func(current), candidate_fitness) > np.random.rand():
            return candidate, candidate_fitness
        return current, func(current)

    def _local_search(self, current, func):
        perturbation = np.random.normal(0, 0.1, size=current.shape)
        candidate = np.clip(current + perturbation, self.bounds.lb, self.bounds.ub)
        return self._anneal(candidate, current, func)

    def _adaptive_parameters(self, evaluations):
        self.f = max(0.5, 1.0 - (evaluations / self.budget))
        self.cr = min(0.9, 0.5 + (evaluations / self.budget))

    def __call__(self, func):
        self.bounds = func.bounds
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                self._adaptive_parameters(evaluations)
                mutant = self._mutate(i, population)
                trial = self._crossover(population[i], mutant)
                population[i], fitness[i] = self._anneal(trial, population[i], func)
                population[i], fitness[i] = self._local_search(population[i], func)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            
            self.temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]