import numpy as np

class ProgressiveMemoryChaoticSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.memory = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = int(np.sqrt(self.budget))
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([self._evaluate(func, ind) for ind in population])

        initial_mutation_rate = 0.15
        mutation_rate = initial_mutation_rate

        scale_factor = 0.1 + 0.9 * np.random.rand()
        while self.evaluations < self.budget:
            memory_impression = self._calculate_memory_impression(fitness)
            mutation_rate = self._adapt_mutation_rate(memory_impression, initial_mutation_rate, fitness)

            for i in range(population_size):
                trial = self._enhanced_chaotic_search(population, fitness, i, lb, ub, func, scale_factor)
                trial_fitness = self._evaluate(func, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                self.memory.append((trial, trial_fitness))

            scale_factor = self._update_scale_factor(scale_factor, memory_impression)

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)

    def _enhanced_chaotic_search(self, population, fitness, index, lb, ub, func, scale_factor):
        best_memory = min(self.memory, key=lambda x: x[1])[0] if self.memory else population[index]
        chaos_factor = np.random.normal(0, scale_factor, size=self.dim)
        perturbation = np.sin(chaos_factor) * scale_factor
        trial = np.clip(best_memory + perturbation, lb, ub)
        return trial

    def _calculate_memory_impression(self, fitness):
        if not self.memory:
            return 0
        memory_fitness = np.array([fit for _, fit in self.memory])
        global_best_memory = np.min(memory_fitness)
        global_worst_memory = np.max(memory_fitness)
        return (global_best_memory - np.mean(memory_fitness)) / (global_worst_memory - global_best_memory + 1e-6)

    def _adapt_mutation_rate(self, memory_impression, initial_mutation_rate, fitness):
        fitness_variance = np.var(fitness)
        return initial_mutation_rate * (1 + 0.5 * memory_impression) * (1 + fitness_variance)

    def _update_scale_factor(self, scale_factor, memory_impression):
        return scale_factor * (0.95 + 0.1 * memory_impression)