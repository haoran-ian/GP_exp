import numpy as np

class AHSO_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.05
        self.c2 = 2.05
        self.inertia_weight = 0.729
        self.mutation_prob = 0.1
        self.crossover_prob = 0.7
        self.elite_fraction = 0.1  # Preserving top 10% as elites

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

            # Update velocities and positions (PSO dynamics)
            # Adaptive inertia weight with dynamic parameter adjustment
            self.inertia_weight = 0.4 + 0.5 * (1 - evaluations / self.budget)
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            velocities = np.clip(velocities, -0.2 * (ub - lb), 0.2 * (ub - lb))  # Velocity clamping
            population = population + velocities
            population = np.clip(population, lb, ub)

            # Genetic Algorithm Crossover and Mutation
            offspring = []
            for _ in range(self.population_size // 2):
                if np.random.rand() < self.crossover_prob:
                    parents = np.random.choice(self.population_size, 2, replace=False)
                    alpha = np.random.uniform(0, 1, self.dim)  # Blend crossover
                    child1 = alpha * population[parents[0]] + (1 - alpha) * population[parents[1]]
                    child2 = alpha * population[parents[1]] + (1 - alpha) * population[parents[0]]
                    offspring.extend([child1, child2])
                else:
                    offspring.extend(population[np.random.choice(self.population_size, 2)])
            
            # Mutation
            offspring = np.array(offspring)
            mutation_mask = np.random.rand(*offspring.shape) < self.mutation_prob
            mutation_values = np.random.normal(0, 0.1 * (ub - lb), offspring.shape)  # Gaussian perturbation
            offspring[mutation_mask] += mutation_values[mutation_mask]

            # Ensure the offspring are within bounds
            offspring = np.clip(offspring, lb, ub)

            # Preserve elites and replace non-elites
            elite_count = int(self.elite_fraction * self.population_size)
            elite_idx = np.argsort(scores)[:elite_count]
            worst_idx = np.argsort(scores)[-self.population_size//2:]
            non_elite_idx = np.setdiff1d(worst_idx, elite_idx, assume_unique=True)
            population[non_elite_idx] = offspring[:len(non_elite_idx)]
            # Elite crossover
            elite_offspring = np.array([population[elite_idx[i % elite_count]] for i in range(len(non_elite_idx))])
            population[non_elite_idx] = 0.5 * population[non_elite_idx] + 0.5 * elite_offspring

            # Local search around global best
            if evaluations + self.population_size <= self.budget:
                local_perturbation = np.random.normal(0, 0.1 * (ub - lb), global_best_position.shape)
                local_search_positions = global_best_position + local_perturbation
                local_search_positions = np.clip(local_search_positions, lb, ub)
                local_scores = np.array([func(pos) for pos in local_search_positions])
                evaluations += len(local_search_positions)
                best_local_idx = np.argmin(local_scores)
                if local_scores[best_local_idx] < global_best_score:
                    global_best_position = local_search_positions[best_local_idx]
                    global_best_score = local_scores[best_local_idx]

        return global_best_position, global_best_score