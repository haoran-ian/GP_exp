import numpy as np

class AHSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.min_inertia_weight = 0.4
        self.max_inertia_weight = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_prob = 0.1
        self.crossover_prob = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        evaluations = 0

        while evaluations < self.budget:
            scores = np.array([func(ind) for ind in population])
            evaluations += len(population)

            # Update personal and global bests
            better_positions = scores < personal_best_scores
            personal_best_positions[better_positions] = population[better_positions]
            personal_best_scores[better_positions] = scores[better_positions]
            
            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_position = personal_best_positions[min_score_idx]
                global_best_score = personal_best_scores[min_score_idx]

            # Adaptive inertia weight
            inertia_weight = self.max_inertia_weight - (evaluations / self.budget) * (self.max_inertia_weight - self.min_inertia_weight)

            # Update velocities and positions (PSO dynamics)
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (inertia_weight * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

            # Genetic Algorithm Crossover and Mutation with crowding
            offspring = []
            crowding_distances = self._calculate_crowding_distances(population)
            for _ in range(self.population_size // 2):
                if np.random.rand() < self.crossover_prob:
                    parents = np.random.choice(self.population_size, 2, replace=False, p=crowding_distances)
                    alpha = np.random.uniform(0, 1, self.dim)  # Blend crossover
                    child1 = alpha * population[parents[0]] + (1 - alpha) * population[parents[1]]
                    child2 = alpha * population[parents[1]] + (1 - alpha) * population[parents[0]]
                    offspring.extend([child1, child2])
                else:
                    offspring.extend(population[np.random.choice(self.population_size, 2)])
            
            # Mutation
            offspring = np.array(offspring)
            mutation_mask = np.random.rand(*offspring.shape) < self.mutation_prob
            mutation_values = np.random.uniform(lb, ub, offspring.shape)
            offspring[mutation_mask] = mutation_values[mutation_mask]

            # Ensure the offspring are within bounds
            offspring = np.clip(offspring, lb, ub)

            # Replace worst half of the population with offspring
            worst_idx = np.argsort(scores)[-self.population_size//2:]
            population[worst_idx] = offspring[:len(worst_idx)]

        return global_best_position, global_best_score

    def _calculate_crowding_distances(self, population):
        # Calculate crowding distances for diversity
        population_size = len(population)
        distances = np.zeros(population_size)
        for i in range(self.dim):
            sorted_indices = np.argsort(population[:, i])
            sorted_pop = population[sorted_indices]
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
            for j in range(1, population_size - 1):
                distances[sorted_indices[j]] += (sorted_pop[j + 1, i] - sorted_pop[j - 1, i]) / (np.max(population[:, i]) - np.min(population[:, i]) + 1e-9)
        return distances / np.sum(distances)