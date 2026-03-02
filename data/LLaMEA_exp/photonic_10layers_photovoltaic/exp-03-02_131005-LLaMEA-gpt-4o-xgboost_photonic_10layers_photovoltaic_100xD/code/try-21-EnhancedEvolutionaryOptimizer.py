import numpy as np

class EnhancedEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Define search space boundaries
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize population
        population_size = int(np.sqrt(self.budget))
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([self._evaluate(func, ind) for ind in population])

        # Strategy parameters
        exploration_factor = 0.7
        exploitation_factor = 0.3
        mutation_rate = 0.2

        # Dual adaptation mechanisms
        adaptive_threshold = 0.5

        while self.evaluations < self.budget:

            # Calculate adaptive weights
            exploration_weight, exploitation_weight = self._dynamic_weights(fitness, adaptive_threshold, exploration_factor, exploitation_factor)

            # Generate offspring using exploration and exploitation
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

            # Adaptive phase based on landscape features
            if self._phase_transition_condition(fitness, adaptive_threshold):
                exploration_factor *= 0.9
                exploitation_factor *= 1.1
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
        perturbation = np.random.laplace(0, mutation_rate, size=self.dim)
        trial = np.clip(individual + perturbation, lb, ub)
        return trial

    def _enhanced_local_search(self, population, fitness, index, lb, ub, func):
        neighbors = self._get_neighbors(population, index)
        neighbor_fitness = [func(neighbor) for neighbor in neighbors]
        best_neighbor = neighbors[np.argmin(neighbor_fitness)]
        weighted_direction = 0.6 * (best_neighbor - population[index])
        trial = np.clip(population[index] + weighted_direction, lb, ub)
        return trial

    def _get_neighbors(self, population, index):
        neighbor_indices = np.random.choice(len(population), min(3, len(population)-1), replace=False)
        neighbors = population[neighbor_indices]
        return neighbors

    def _dynamic_weights(self, fitness, threshold, exploration_factor, exploitation_factor):
        global_best = np.min(fitness)
        global_impression = (global_best - np.mean(fitness)) / (np.ptp(fitness) + 1e-6)
        exploration_weight = exploration_factor * (1 - global_impression)
        exploitation_weight = exploitation_factor * global_impression
        return exploration_weight, exploitation_weight

    def _phase_transition_condition(self, fitness, threshold):
        sorted_fitness = np.sort(fitness)
        phase_threshold = np.percentile(sorted_fitness, threshold * 100)
        return np.any(sorted_fitness[:int(threshold * len(fitness))] < phase_threshold)