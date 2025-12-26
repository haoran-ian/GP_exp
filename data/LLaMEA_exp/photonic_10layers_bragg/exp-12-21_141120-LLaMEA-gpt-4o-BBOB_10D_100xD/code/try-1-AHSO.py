import numpy as np

class AHSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.05
        self.c2 = 2.05
        self.inertia_weight = 0.9  # Adjusted initial inertia weight
        self.final_inertia_weight = 0.4  # New final inertia weight
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

            # Dynamic inertia weight adjustment
            self.inertia_weight = (self.final_inertia_weight +
                                   (0.9 - self.final_inertia_weight) *
                                   ((self.budget - evaluations) / self.budget))

            # Update velocities and positions (PSO dynamics)
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

            # Genetic Algorithm Crossover and Mutation
            offspring = []
            elite_size = self.population_size // 10  # New elite selection
            elite_indices = np.argsort(scores)[:elite_size]
            for _ in range(self.population_size // 2):
                if np.random.rand() < self.crossover_prob:
                    parents = np.random.choice(self.population_size, 2, replace=False)
                    cutpoint = np.random.randint(1, self.dim - 1)
                    child1 = np.concatenate((population[parents[0], :cutpoint],
                                             population[parents[1], cutpoint:]))
                    child2 = np.concatenate((population[parents[1], :cutpoint],
                                             population[parents[0], cutpoint:]))
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

            # Replace worst half of the population with offspring, preserving elites
            worst_idx = np.argsort(scores)[-self.population_size//2:]
            non_elite_worst_idx = worst_idx[~np.in1d(worst_idx, elite_indices)]
            population[non_elite_worst_idx] = offspring[:len(non_elite_worst_idx)]

        return global_best_position, global_best_score