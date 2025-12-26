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

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / beta)
        return step

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

            better_positions = scores < personal_best_scores
            personal_best_positions[better_positions] = population[better_positions]
            personal_best_scores[better_positions] = scores[better_positions]
            
            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_position = personal_best_positions[min_score_idx]
                global_best_score = personal_best_scores[min_score_idx]

            self.inertia_weight = 0.4 + 0.5 * evaluations / self.budget
            adaptive_c1 = self.c1 - (self.c1 - 1.5) * evaluations / self.budget
            adaptive_c2 = self.c2 + (2.5 - self.c2) * evaluations / self.budget
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          adaptive_c1 * r1 * (personal_best_positions - population) +
                          adaptive_c2 * r2 * (global_best_position - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

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
            
            offspring = np.array(offspring)
            mutation_mask = np.random.rand(*offspring.shape) < self.mutation_prob
            mutation_values = np.random.uniform(lb, ub, offspring.shape)
            offspring[mutation_mask] = mutation_values[mutation_mask]

            # Introduce Lévy flight for further exploration
            levy_steps = self.levy_flight(offspring.shape)
            offspring = offspring + levy_steps * (ub - lb) * 0.01
            offspring = np.clip(offspring, lb, ub)

            worst_idx = np.argsort(scores)[-self.population_size//2:]
            population[worst_idx] = offspring[:len(worst_idx)]

        return global_best_position, global_best_score