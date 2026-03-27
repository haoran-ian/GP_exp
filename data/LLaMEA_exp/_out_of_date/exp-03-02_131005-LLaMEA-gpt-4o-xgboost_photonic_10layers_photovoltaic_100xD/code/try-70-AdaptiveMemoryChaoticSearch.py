import numpy as np

class AdaptiveMemoryChaoticSearch:
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
        
        exploration_factor = 0.6
        exploitation_factor = 0.4
        mutation_rate = 0.15

        while self.evaluations < self.budget:
            memory_impression = self._calculate_memory_impression(fitness)
            exploration_weight = exploration_factor * (1 - memory_impression)
            exploitation_weight = exploitation_factor * memory_impression

            exploration_agents = np.random.choice(range(population_size), size=int(population_size * exploration_weight), replace=False)
            exploitation_agents = np.setdiff1d(range(population_size), exploration_agents)

            for i in exploration_agents:
                trial, elitism_factor = self._memory_chaotic_perturbation(population[i], lb, ub, mutation_rate)
                trial_fitness = self._evaluate(func, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                self.memory.append((trial, trial_fitness))

            for i in exploitation_agents:
                trial = self._chaos_driven_local_search(population, fitness, i, lb, ub, func)
                trial_fitness = self._evaluate(func, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            if len(self.memory) > 10:
                past_fitness = np.array([fit for _, fit in self.memory[-10:]])
                mutation_rate *= (1 - 0.3 * np.var(past_fitness) / np.mean(past_fitness))

            if self.evaluations >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)

    def _memory_chaotic_perturbation(self, individual, lb, ub, mutation_rate):
        if self.memory:
            best_memory_ind = min(self.memory, key=lambda x: x[1])[0]
        else:
            best_memory_ind = individual
        chaos_factor = np.random.normal(0, mutation_rate * 1.5, size=self.dim)
        perturbation = np.sin(chaos_factor) * mutation_rate
        elitism_factor = 0.3
        trial = np.clip(best_memory_ind + perturbation * elitism_factor, lb, ub)
        return trial, elitism_factor

    def _chaos_driven_local_search(self, population, fitness, index, lb, ub, func):
        neighbors = self._get_neighbors(population, index)
        best_neighbor = min(neighbors, key=lambda ind: func(ind))
        weighted_direction = 0.7 * (best_neighbor - population[index])
        chaos_direction = np.sin(weighted_direction) * 0.25
        trial = np.clip(population[index] + chaos_direction, lb, ub)
        return trial

    def _get_neighbors(self, population, index):
        neighbor_indices = np.random.choice(len(population), min(3, len(population)-1), replace=False)
        neighbors = population[neighbor_indices]
        return neighbors

    def _calculate_memory_impression(self, fitness):
        if not self.memory:
            return 0
        memory_fitness = np.array([fit for _, fit in self.memory])
        global_best_memory = np.min(memory_fitness)
        global_worst_memory = np.max(memory_fitness)
        return (global_best_memory - np.mean(memory_fitness)) / (global_worst_memory - global_best_memory + 1e-6)