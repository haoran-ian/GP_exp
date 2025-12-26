import numpy as np

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.current_population_size = self.initial_population_size
        self.initial_temp = 100.0
        self.cooling_rate = 0.97  # Further refined cooling rate for gradual exploration-exploitation balance
        self.bounds = None
        self.elitism_rate = 0.2
        self.dynamic_population_scale = 0.5  # Rate to scale the population dynamically

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.current_population_size, self.dim))

    def _adaptive_mutation(self, fitness, f_min=0.3, f_max=0.8):  # Refined mutation adaptability
        norm_fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-6)
        return f_min + (f_max - f_min) * (1 - norm_fitness)

    def _mutate(self, target_idx, population, fitness):
        indices = [idx for idx in range(self.current_population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = self._adaptive_mutation(fitness[target_idx])
        mutant = population[a] + f * (population[b] - population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _adaptive_crossover(self, generation_ratio):
        return 0.9 - 0.4 * generation_ratio  # Dynamic crossover rate based on progress

    def _crossover(self, target, mutant, generation_ratio):
        cr = self._adaptive_crossover(generation_ratio)
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
        evaluations = self.current_population_size
        temperature = self.initial_temp
        elite_size = int(self.elitism_rate * self.current_population_size)

        generation = 0
        while evaluations < self.budget:
            generation_ratio = evaluations / self.budget
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            for i in range(self.current_population_size):
                if i < elite_size:
                    continue
                mutant = self._mutate(i, population, fitness)
                trial = self._crossover(population[i], mutant, generation_ratio)
                new_population[i], new_fitness[i] = self._anneal(trial, population[i], func, temperature)
                evaluations += 1
                if evaluations >= self.budget:
                    break

            combined_pop = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.current_population_size]
            population = combined_pop[best_indices]
            fitness = combined_fitness[best_indices]

            temperature *= self.cooling_rate

            # Dynamic population resizing
            if evaluations < self.budget * 0.5:
                self.current_population_size = int(self.initial_population_size * (1 + self.dynamic_population_scale * (1 - generation_ratio)))

            generation += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]