import numpy as np

class DynamicAnnealingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_temp = 100.0
        self.cooling_rate = 0.98
        self.bounds = None
        self.elitism_rate = 0.2
        self.trend_sensitivity = 0.1  # New parameter for adaptive cooling
        self.improvement_threshold = 1e-6  # Threshold to detect fitness improvement trend

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _adaptive_mutation(self, fitness, f_min=0.5, f_max=0.9):
        norm_fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-6)
        return f_min + (f_max - f_min) * (1 - norm_fitness)

    def _mutate(self, target_idx, population, fitness):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = self._adaptive_mutation(fitness[target_idx])
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

    def _adaptive_cooling(self, previous_best_fitness, current_best_fitness, temperature):
        if current_best_fitness < previous_best_fitness - self.improvement_threshold:
            return temperature * (1 + self.trend_sensitivity)
        return temperature * self.cooling_rate

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
        previous_best_fitness = np.min(fitness)

        while evaluations < self.budget:
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            for i in range(self.population_size):
                if i < elite_size:
                    continue
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

            current_best_fitness = np.min(fitness)
            temperature = self._adaptive_cooling(previous_best_fitness, current_best_fitness, temperature)
            previous_best_fitness = current_best_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]