import numpy as np

class EnhancedDynamicChaoticSearch:
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

        exploration_factor = 0.7
        exploitation_factor = 0.3
        initial_mutation_rate = 0.1
        mutation_rate = initial_mutation_rate

        while self.evaluations < self.budget:
            memory_impression = self._calculate_memory_impression(fitness)
            exploration_weight = exploration_factor * (1 - memory_impression)
            exploitation_weight = exploitation_factor * memory_impression

            exploration_agents = np.random.choice(range(population_size), 
                                                  size=int(population_size * exploration_weight), 
                                                  replace=False)
            exploitation_agents = np.setdiff1d(range(population_size), exploration_agents)

            for i in exploration_agents:
                trial = self._dynamic_chaotic_perturbation(population[i], lb, ub, mutation_rate)
                trial_fitness = self._evaluate(func, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                self.memory.append((trial, trial_fitness))

            for i in exploitation_agents:
                trial = self._elite_guided_exploitation(population, fitness, i, lb, ub, func)
                trial_fitness = self._evaluate(func, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            mutation_rate = self._adapt_mutation_rate(memory_impression, initial_mutation_rate, fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)

    def _dynamic_chaotic_perturbation(self, individual, lb, ub, mutation_rate):
        chaos_factor = np.random.normal(0, mutation_rate * np.sqrt(self.evaluations/self.budget), size=self.dim)
        perturbation = np.sin(chaos_factor) * mutation_rate
        trial = np.clip(individual + perturbation, lb, ub)
        return trial

    def _elite_guided_exploitation(self, population, fitness, index, lb, ub, func):
        elite_idx = np.argmin(fitness)
        elite_direction = population[elite_idx] - population[index]
        weighted_direction = 0.8 * elite_direction
        chaos_direction = np.sin(weighted_direction) * 0.2
        trial = np.clip(population[index] + chaos_direction, lb, ub)
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
        return initial_mutation_rate * (1 + memory_impression) * (1 + fitness_variance)