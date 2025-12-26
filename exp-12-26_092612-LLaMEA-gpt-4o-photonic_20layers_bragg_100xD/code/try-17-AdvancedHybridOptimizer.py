import numpy as np

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.max_population_size = 50  # Allows dynamic population growth
        self.initial_temp = 100.0
        self.cooling_rate = 0.95  # Adjusted for strategic diversification
        self.bounds = None
        self.elitism_rate = 0.2

    def _initialize_population(self, size):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (size, self.dim))

    def _dynamic_population_adjustment(self, current_gen):
        # Increasing population for better exploration in later stages
        return min(self.initial_population_size + current_gen * 2, self.max_population_size)

    def _adaptive_mutation(self, fitness, f_min=0.3, f_max=0.8):
        norm_fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-6)
        return f_min + (f_max - f_min) * (1 - norm_fitness)

    def _mutate(self, target_idx, population, fitness):
        indices = [idx for idx in range(len(population)) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = self._adaptive_mutation(fitness[target_idx])
        mutant = population[a] + f * (population[b] - population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _crossover(self, target, mutant, cr=0.6):
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
        population_size = self.initial_population_size
        population = self._initialize_population(population_size)
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        temperature = self.initial_temp
        generation = 0

        while evaluations < self.budget:
            new_population_size = self._dynamic_population_adjustment(generation)
            new_population = self._initialize_population(new_population_size)
            new_fitness = np.full(new_population_size, np.inf)
            elite_size = int(self.elitism_rate * new_population_size)

            for i in range(new_population_size):
                if i < elite_size and i < len(population):
                    new_population[i] = population[np.argsort(fitness)[:elite_size]][i]
                    new_fitness[i] = fitness[np.argsort(fitness)[:elite_size]][i]
                    continue
                if evaluations >= self.budget:
                    break
                mutant = self._mutate(i % len(population), population, fitness)
                trial = self._crossover(population[i % len(population)], mutant)
                new_population[i], new_fitness[i] = self._anneal(trial, population[i % len(population)], func, temperature)
                evaluations += 1

            combined_pop = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:new_population_size]
            population = combined_pop[best_indices]
            fitness = combined_fitness[best_indices]

            temperature *= self.cooling_rate
            generation += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]