import numpy as np

class EnhancedHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1_initial = 1.5
        self.c2_initial = 1.5
        self.w_initial = 0.9
        self.w_final = 0.4
        self.mutation_rate_initial = 0.1
        self.mutation_rate_final = 0.01
        self.cross_over_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.w = self.w_initial - progress * (self.w_initial - self.w_final)
            self.c1 = self.c1_initial + progress * (2.0 - self.c1_initial)
            self.c2 = self.c2_initial + progress * (2.0 - self.c2_initial)
            self.mutation_rate = self.mutation_rate_initial - progress * (self.mutation_rate_initial - self.mutation_rate_final)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best - population) +
                          self.c2 * r2 * (global_best - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            improved = scores < personal_best_scores
            personal_best[improved] = population[improved]
            personal_best_scores[improved] = scores[improved]
            if min(scores) < global_best_score:
                global_best = population[np.argmin(scores)]
                global_best_score = min(scores)

            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.cross_over_rate:
                    cross_point = np.random.randint(1, self.dim)
                    parent1, parent2 = population[i], population[i + 1]
                    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                    population[i], population[i + 1] = child1, child2

            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_rate
            mutation_values = np.random.uniform(lb, ub, (self.population_size, self.dim))
            population = np.where(mutation_mask, mutation_values, population)

        return global_best, global_best_score