import numpy as np

class AHSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5 + np.random.rand() * 1.5  # Adaptive
        self.c2 = 1.5 + np.random.rand() * 1.5  # Adaptive
        self.inertia_weight = 0.5 + np.random.rand() * 0.5  # Adaptive
        self.mutation_prob = 0.2  # Increased
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

            # Update velocities and positions (PSO dynamics)
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

            # Genetic Algorithm Crossover and Mutation
            offspring = []
            for _ in range(self.population_size // 2):
                if np.random.rand() < self.crossover_prob:
                    parents = np.random.choice(self.population_size, 2, replace=False)
                    alpha = np.random.uniform(0, 1, self.dim)  
                    child1 = alpha * population[parents[0]] + (1 - alpha) * population[parents[1]]
                    child2 = alpha * population[parents[1]] + (1 - alpha) * population[parents[0]]
                    offspring.extend([child1, child2])
                else:
                    offspring.extend(population[np.random.choice(self.population_size, 2)])

            # Mutation
            offspring = np.array(offspring)
            mutation_mask = np.random.rand(*offspring.shape) < self.mutation_prob
            gaussian_mutation = np.random.normal(0, 0.1, offspring.shape)  # Gaussian mutation
            offspring[mutation_mask] += gaussian_mutation[mutation_mask]

            # Ensure the offspring are within bounds
            offspring = np.clip(offspring, lb, ub)

            # Replace worst half of the population with offspring
            worst_idx = np.argsort(scores)[-self.population_size//2:]
            population[worst_idx] = offspring[:len(worst_idx)]

        return global_best_position, global_best_score