import numpy as np

class ImprovedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_temp = 100.0
        self.cooling_rate = 0.95
        self.bounds = None
        self.mutation_memory = [0.5, 0.9]  # Track successful mutation factors

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _mutate(self, target_idx, population):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = np.random.choice(self.mutation_memory)  # Use self-adaptive mutation factor
        mutant = population[a] + f * (population[b] - population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _crossover(self, target, mutant, cr):
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
            cr = 0.5 * (1 + np.tanh((self.initial_temp - temperature) / 50))  # Temperature-dependent crossover rate
            for i in range(self.population_size):
                mutant = self._mutate(i, population)
                trial = self._crossover(population[i], mutant, cr)
                old_fitness = fitness[i]
                population[i], fitness[i] = self._anneal(trial, population[i], func, temperature)
                if fitness[i] < old_fitness:  # Update mutation memory
                    self.mutation_memory.append(np.random.uniform(0.5, 0.9))
                evaluations += 1
                if evaluations >= self.budget:
                    break
            
            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]