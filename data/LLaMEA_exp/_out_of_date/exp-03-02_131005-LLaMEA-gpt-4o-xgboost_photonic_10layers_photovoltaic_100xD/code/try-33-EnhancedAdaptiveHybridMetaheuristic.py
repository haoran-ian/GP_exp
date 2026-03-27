import numpy as np

class EnhancedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = int(np.sqrt(self.budget))
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([self._evaluate(func, ind) for ind in population])

        exploration_factor = 0.6
        exploitation_factor = 0.4
        mutation_rate = 0.1
        
        chaos_control_param = 1.0
        mutation_adaptation_rate = 0.05

        while self.evaluations < self.budget:
            global_impression = self._calculate_global_impression(fitness)
            exploration_weight = exploration_factor * (1 - global_impression)
            exploitation_weight = exploitation_factor * global_impression

            offspring = []
            for i in range(population_size):
                if np.random.rand() < exploration_weight:
                    trial = self._chaotic_random_perturbation(population[i], lb, ub, mutation_rate, chaos_control_param)
                else:
                    trial = self._chaos_driven_local_search(population, fitness, i, lb, ub, func)

                trial_fitness = self._evaluate(func, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                offspring.append((trial, trial_fitness))

            if self._phase_transition_condition(fitness):
                exploration_factor = max(0.1, exploration_factor * 0.87)
                exploitation_factor = min(0.9, exploitation_factor * 1.13)
                mutation_rate = max(0.01, mutation_rate * (1 - mutation_adaptation_rate))

            chaos_control_param = self._adjust_chaos_control_param(fitness)

            if self.evaluations >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)

    def _chaotic_random_perturbation(self, individual, lb, ub, mutation_rate, chaos_control_param):
        chaos_factor = np.random.normal(0, mutation_rate * 1.1 * chaos_control_param, size=self.dim)
        perturbation = np.sin(chaos_factor) * mutation_rate
        return np.clip(individual + perturbation, lb, ub)

    def _chaos_driven_local_search(self, population, fitness, index, lb, ub, func):
        neighbors = self._get_neighbors(population, index)
        best_neighbor = min(neighbors, key=lambda ind: func(ind))
        weighted_direction = 0.5 * (best_neighbor - population[index])
        chaos_direction = np.sin(weighted_direction) * 0.1
        return np.clip(population[index] + chaos_direction, lb, ub)

    def _get_neighbors(self, population, index):
        neighbor_indices = np.random.choice(len(population), min(3, len(population)-1), replace=False)
        return population[neighbor_indices]

    def _calculate_global_impression(self, fitness):
        global_best = np.min(fitness)
        global_worst = np.max(fitness)
        return (global_best - np.mean(fitness)) / (global_worst - global_best + 1e-6)

    def _phase_transition_condition(self, fitness):
        sorted_fitness = np.sort(fitness)
        phase_threshold = np.percentile(sorted_fitness, 15)
        return np.any(sorted_fitness[:int(0.15 * len(fitness))] < phase_threshold)

    def _adjust_chaos_control_param(self, fitness):
        variance = np.var(fitness)
        return 1.0 + np.tanh(variance)