import numpy as np

class EnhancedAHSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.05
        self.c2 = 2.05
        self.base_inertia_weight = 0.729
        self.mutation_prob = 0.1
        self.crossover_prob = 0.7
        self.subpop_size = 10

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

            # Update velocities and positions with adaptive inertia weight
            self.inertia_weight = self.base_inertia_weight * (1 - evaluations / self.budget)
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

            # Regroup subpopulations dynamically
            np.random.shuffle(population)
            for i in range(0, self.population_size, self.subpop_size):
                subpop = population[i:i+self.subpop_size]
                subpop_scores = scores[i:i+self.subpop_size]
                worst_idx = np.argsort(subpop_scores)[-self.subpop_size//2:]
                best_idx = np.argsort(subpop_scores)[:self.subpop_size//2]

                # Crossover within subpopulation best individuals
                for wi, bi in zip(worst_idx, best_idx):
                    if np.random.rand() < self.crossover_prob:
                        alpha = np.random.uniform(0, 1, self.dim)
                        subpop[wi] = alpha * subpop[wi] + (1 - alpha) * subpop[bi]
                
                # Mutation
                mutation_mask = np.random.rand(*subpop.shape) < self.mutation_prob
                mutation_values = np.random.uniform(lb, ub, subpop.shape)
                subpop[mutation_mask] = mutation_values[mutation_mask]

                # Update subpopulation in the main population
                population[i:i+self.subpop_size] = subpop

        return global_best_position, global_best_score