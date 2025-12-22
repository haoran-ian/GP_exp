import numpy as np

class EnhancedAdaptiveHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_population_size = 50
        self.min_population_size = 20
        self.max_population_size = 100
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.w_initial = 0.9
        self.w_final = 0.4
        self.mutation_rate_initial = 0.2
        self.mutation_rate_final = 0.05
        self.cross_over_rate = 0.7
        self.local_search_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.init_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = population_size
        t = 0

        while evaluations < self.budget:
            w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)
            mutation_rate = self.mutation_rate_initial - (self.mutation_rate_initial - self.mutation_rate_final) * (evaluations / self.budget)

            r1, r2 = np.random.rand(population_size, self.dim), np.random.rand(population_size, self.dim)
            velocities = (w * velocities + 
                          self.c1_initial * r1 * (personal_best - population) + 
                          self.c2_initial * r2 * (global_best - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

            scores = np.array([func(ind) for ind in population])
            evaluations += population_size

            improved = scores < personal_best_scores
            personal_best[improved] = population[improved]
            personal_best_scores[improved] = scores[improved]
            if min(scores) < global_best_score:
                global_best = population[np.argmin(scores)]
                global_best_score = min(scores)

            for i in range(0, population_size, 2):
                if np.random.rand() < self.cross_over_rate:
                    cross_point = np.random.randint(1, self.dim)
                    parent1, parent2 = population[i], population[i + 1]
                    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                    population[i], population[i + 1] = child1, child2

            mutation_mask = np.random.rand(population_size, self.dim) < mutation_rate
            mutation_values = np.random.uniform(lb, ub, (population_size, self.dim))
            population = np.where(mutation_mask, mutation_values, population)

            if np.random.rand() < self.local_search_rate:
                for i in range(population_size):
                    if np.random.rand() < self.local_search_rate:
                        perturbation = np.random.normal(0, 0.1, self.dim)
                        candidate = np.clip(population[i] + perturbation, lb, ub)
                        candidate_score = func(candidate)
                        evaluations += 1
                        if candidate_score < scores[i]:
                            population[i] = candidate
                            scores[i] = candidate_score

            population_size = min(max(self.min_population_size, population_size + int(np.random.normal(0, 5))), self.max_population_size)
            population = population[:population_size, :]
            velocities = velocities[:population_size, :]
            personal_best = personal_best[:population_size, :]
            personal_best_scores = personal_best_scores[:population_size]

            t += 1

        return global_best, global_best_score