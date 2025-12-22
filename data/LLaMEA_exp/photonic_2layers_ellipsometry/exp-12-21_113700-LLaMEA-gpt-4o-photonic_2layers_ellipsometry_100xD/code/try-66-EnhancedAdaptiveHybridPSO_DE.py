import numpy as np

class EnhancedAdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.w_initial = 0.9
        self.w_final = 0.4
        self.mutation_rate_initial = 0.2
        self.mutation_rate_final = 0.05
        self.cross_over_rate = 0.7
        self.de_factor = 0.8  # Differential Evolution factor
        self.learning_factor = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size
        t = 0

        while evaluations < self.budget:
            w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)
            mutation_rate = self.mutation_rate_initial - (self.mutation_rate_initial - self.mutation_rate_final) * (evaluations / self.budget)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          self.c1_initial * r1 * (personal_best - population) + 
                          self.c2_initial * r2 * (global_best - population))
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

            # Differential Evolution Strategy
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant_vector = x1 + self.de_factor * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, lb, ub)
                cross_points = np.random.rand(self.dim) < self.cross_over_rate
                trial_vector = np.where(cross_points, mutant_vector, population[i])
                if func(trial_vector) < scores[i]:
                    population[i] = trial_vector
                    scores[i] = func(trial_vector)

            t += 1

        return global_best, global_best_score