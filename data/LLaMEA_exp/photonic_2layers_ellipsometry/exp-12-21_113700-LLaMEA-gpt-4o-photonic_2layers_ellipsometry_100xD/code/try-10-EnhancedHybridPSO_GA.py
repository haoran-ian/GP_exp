import numpy as np

class EnhancedHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5  # cognitive parameter
        self.c2 = 1.5  # social parameter
        self.w = 0.5   # inertia weight
        self.mutation_rate = 0.1
        self.cross_over_rate = 0.7
        self.f = 0.5    # differential weight
        self.cr = 0.9   # crossover probability

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
            # Dynamically adjust parameters
            self.w = 0.9 - 0.4 * (evaluations / self.budget)
            self.c1 = 1.5 + (evaluations / self.budget)

            # Update velocities and population positions (PSO Step)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (personal_best - population) + 
                          self.c2 * r2 * (global_best - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

            # Evaluate the population
            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            # Update personal best and global best
            improved = scores < personal_best_scores
            personal_best[improved] = population[improved]
            personal_best_scores[improved] = scores[improved]
            if min(scores) < global_best_score:
                global_best = population[np.argmin(scores)]
                global_best_score = min(scores)

            # Apply Genetic Operators (GA Step)
            # Crossover
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.cross_over_rate:
                    cross_point = np.random.randint(1, self.dim)
                    parent1, parent2 = population[i], population[i + 1]
                    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                    population[i], population[i + 1] = child1, child2

            # Mutation
            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_rate
            mutation_values = np.random.uniform(lb, ub, (self.population_size, self.dim))
            population = np.where(mutation_mask, mutation_values, population)

            # Differential Evolution Step
            for i in range(self.population_size):
                idxs = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(a + self.f * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_score = func(trial)
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

        return global_best, global_best_score