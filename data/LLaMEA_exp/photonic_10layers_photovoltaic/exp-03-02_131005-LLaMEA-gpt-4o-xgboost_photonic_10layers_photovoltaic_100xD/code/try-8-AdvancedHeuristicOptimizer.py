import numpy as np

class AdvancedHeuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        population_size = int(np.sqrt(self.budget))
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([self._evaluate(func, ind) for ind in population])

        exploration_factor, exploitation_factor = 0.7, 0.3

        while self.evaluations < self.budget:
            global_impression = self._calculate_global_impression(fitness)
            exploration_weight = exploration_factor * (1 - global_impression)
            exploitation_weight = exploitation_factor * global_impression

            partition_size = self._adaptive_partition(population, fitness)
            partitioned_population = np.array_split(population, partition_size)
            partitioned_fitness = np.array_split(fitness, partition_size)

            for part_pop, part_fit in zip(partitioned_population, partitioned_fitness):
                for i in range(len(part_pop)):
                    if np.random.rand() < exploration_weight:
                        trial = self._random_perturbation(part_pop[i], lb, ub)
                    else:
                        trial = self._adaptive_local_search(part_pop, part_fit, i, lb, ub, func)

                    trial_fitness = self._evaluate(func, trial)
                    if trial_fitness < part_fit[i]:
                        part_pop[i] = trial
                        part_fit[i] = trial_fitness

            population = np.concatenate(partitioned_population)
            fitness = np.concatenate(partitioned_fitness)

            if self._phase_transition_condition(fitness):
                exploration_factor *= 0.9
                exploitation_factor *= 1.1

            if self.evaluations >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)

    def _random_perturbation(self, individual, lb, ub):
        perturbation = np.random.normal(0, 0.1, size=self.dim)
        trial = np.clip(individual + perturbation, lb, ub)
        return trial

    def _adaptive_local_search(self, population, fitness, index, lb, ub, func):
        neighbors = self._get_neighbors(population, index)
        best_neighbor = min(neighbors, key=lambda ind: func(ind))
        direction = best_neighbor - population[index]
        trial = np.clip(population[index] + direction, lb, ub)
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
        phase_threshold = np.percentile(sorted_fitness, 10)
        return np.any(sorted_fitness[:int(0.1 * len(fitness))] < phase_threshold)

    def _adaptive_partition(self, population, fitness):
        fitness_range = np.ptp(fitness)
        diversity = np.mean([np.linalg.norm(a-b) for a in population for b in population])
        return max(1, int(diversity / (fitness_range + 1e-6) * 10))