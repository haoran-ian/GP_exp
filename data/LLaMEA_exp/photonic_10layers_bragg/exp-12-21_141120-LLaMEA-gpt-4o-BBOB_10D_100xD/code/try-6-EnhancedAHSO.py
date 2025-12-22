import numpy as np

class EnhancedAHSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.05
        self.c2 = 2.05
        self.inertia_weight = 0.729
        self.mutation_prob = 0.1
        self.crossover_prob = 0.7
        self.f = 0.8  # Differential evolution scaling factor
        self.cr = 0.9  # Differential evolution crossover probability

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

            # Differential Evolution (DE) - Mutation and Crossover
            offspring = []
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + self.f * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)
                trial = np.array([mutant[j] if np.random.rand() < self.cr else population[i, j] for j in range(self.dim)])
                offspring.append(trial)
            
            # Evaluate offspring
            offspring_scores = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            # Select the best individuals
            for i in range(self.population_size):
                if offspring_scores[i] < scores[i]:
                    population[i] = offspring[i]
                    scores[i] = offspring_scores[i]

        return global_best_position, global_best_score