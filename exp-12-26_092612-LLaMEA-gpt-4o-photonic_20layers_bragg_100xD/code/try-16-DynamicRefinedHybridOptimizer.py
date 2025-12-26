import numpy as np

class DynamicRefinedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.bounds = None
        self.initial_temp = 100.0
        self.cooling_rate = 0.98
        self.elitism_rate = 0.2
        self.min_population_size = 10
        self.max_population_size = 50

    def _initialize_population(self, size):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (size, self.dim))

    def _adaptive_mutation(self, fitness, f_min=0.4, f_max=0.9):
        norm_fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-6)
        return f_min + (f_max - f_min) * (1 - norm_fitness)

    def _mutate(self, target_idx, population, fitness):
        indices = [idx for idx in range(len(population)) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = self._adaptive_mutation(fitness[target_idx])
        mutant = population[a] + f * (population[b] - population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _adaptive_crossover_rate(self, generation):
        return 0.1 + 0.9 * (1 - np.exp(-0.05 * generation))

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
        population_size = self.initial_population_size
        population = self._initialize_population(population_size)
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        temperature = self.initial_temp
        generation = 0

        while evaluations < self.budget:
            generation += 1
            cr = self._adaptive_crossover_rate(generation)
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            elite_size = int(self.elitism_rate * population_size)
            for i in range(population_size):
                if i < elite_size:
                    continue
                mutant = self._mutate(i, population, fitness)
                trial = self._crossover(population[i], mutant, cr)
                new_population[i], new_fitness[i] = self._anneal(trial, population[i], func, temperature)
                evaluations += 1
                if evaluations >= self.budget:
                    break

            combined_pop = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:population_size]
            population = combined_pop[best_indices]
            fitness = combined_fitness[best_indices]

            temperature *= self.cooling_rate

            # Dynamically adjust population size
            if generation % 10 == 0:
                population_size = np.random.randint(self.min_population_size, self.max_population_size)
                if population_size > len(population):
                    additional_population = self._initialize_population(population_size - len(population))
                    population = np.vstack((population, additional_population))
                    additional_fitness = np.array([func(ind) for ind in additional_population])
                    fitness = np.hstack((fitness, additional_fitness))
                    evaluations += len(additional_population)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]