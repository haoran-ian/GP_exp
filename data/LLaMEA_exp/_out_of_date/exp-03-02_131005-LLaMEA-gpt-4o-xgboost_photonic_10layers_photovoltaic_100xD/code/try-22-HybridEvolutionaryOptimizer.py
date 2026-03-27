import numpy as np

class HybridEvolutionaryOptimizer:
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

        while self.evaluations < self.budget:
            global_impression = self._calculate_global_impression(fitness)
            exploration_weight = exploration_factor * (1 - global_impression)
            exploitation_weight = exploitation_factor * global_impression

            offspring = []
            for i in range(population_size):
                if np.random.rand() < exploration_weight:
                    trial = self._adaptive_random_perturbation(population[i], lb, ub, mutation_rate)
                else:
                    trial = self._enhanced_local_search(population, fitness, i, lb, ub, func)

                trial_fitness = self._evaluate(func, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                offspring.append((trial, trial_fitness))

            if self._phase_transition_condition(fitness):
                exploration_factor *= 0.85
                exploitation_factor *= 1.15
                mutation_rate *= 0.95

            if self.evaluations >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)

    def _adaptive_random_perturbation(self, individual, lb, ub, mutation_rate):
        perturbation = np.random.normal(0, mutation_rate, size=self.dim)
        trial = np.clip(individual + perturbation, lb, ub)
        return trial

    def _enhanced_local_search(self, population, fitness, index, lb, ub, func):
        neighbors = self._get_neighbors(population, index)
        best_neighbor = min(neighbors, key=lambda ind: func(ind))
        weighted_direction = 0.5 * (best_neighbor - population[index])
        trial = np.clip(population[index] + weighted_direction, lb, ub)
        return trial

    def _get_neighbors(self, population, index):
        neighbor_indices = np.random.choice(len(population), min(3, len(population)-1), replace=False)
        neighbors = population[neighbor_indices]
        return neighbors

    def _calculate_global_impression(self, fitness):
        global_best = np.min(fitness)
        global_worst = np.max(fitness)
        return (global_best - np.mean(fitness)) / (global_worst - global_best + 1e-6)

    def _phase_transition_condition(self, fitness):
        sorted_fitness = np.sort(fitness)
        phase_threshold = np.percentile(sorted_fitness, 15)
        return np.any(sorted_fitness[:int(0.15 * len(fitness))] < phase_threshold)