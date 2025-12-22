import numpy as np

class AHSO_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1_init = 2.5
        self.c2_init = 0.5
        self.inertia_weight = 0.729
        self.mutation_prob = 0.3  # Increased mutation probability
        self.crossover_prob = 0.7
        self.elite_fraction = 0.1
        self.alpha = 1.5

    def levy_flight(self, size):
        sigma = (np.math.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) /
                 (np.math.gamma((1 + self.alpha) / 2) * self.alpha * 2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / self.alpha)
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

            # Adaptive learning factors
            self.c1 = self.c1_init * (1 - evaluations / self.budget)
            self.c2 = self.c2_init * (evaluations / self.budget)

            self.inertia_weight = 0.729 * (0.4 ** (evaluations / self.budget))
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            velocities = np.clip(velocities, -0.2 * (ub - lb), 0.2 * (ub - lb))
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
            dynamic_mutation_prob = self.mutation_prob * (1 - evaluations / self.budget)
            mutation_mask = np.random.rand(*offspring.shape) < dynamic_mutation_prob
            levy_steps = self.levy_flight(offspring.shape)
            offspring[mutation_mask] += levy_steps[mutation_mask]

            offspring = np.clip(offspring, lb, ub)

            elite_count = int(self.elite_fraction * self.population_size)
            elite_idx = np.argsort(scores)[:elite_count]
            worst_idx = np.argsort(scores)[-self.population_size//2:]
            non_elite_idx = np.setdiff1d(worst_idx, elite_idx, assume_unique=True)
            population[non_elite_idx] = offspring[:len(non_elite_idx)]
            elite_offspring = np.array([population[elite_idx[i % elite_count]] for i in range(len(non_elite_idx))])
            population[non_elite_idx] = 0.5 * population[non_elite_idx] + 0.5 * elite_offspring

        return global_best_position, global_best_score