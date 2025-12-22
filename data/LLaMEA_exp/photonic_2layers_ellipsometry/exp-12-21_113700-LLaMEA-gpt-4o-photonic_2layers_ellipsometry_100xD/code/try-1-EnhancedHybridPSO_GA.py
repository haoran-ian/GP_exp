import numpy as np

class EnhancedHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
        self.w = 0.9   # inertia weight
        self.mutation_rate = 0.1
        self.cross_over_rate = 0.7
        self.f = 0.8  # differential weight for mutation in DE

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
            # Adjust parameters dynamically
            self.w = 0.9 - (0.5 * evaluations / self.budget)
            self.c1 = 1.5 + (0.5 * evaluations / self.budget)
            self.c2 = 1.5 - (0.5 * evaluations / self.budget)

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

            # Apply Genetic Operators (GA Step with DE Crossover)
            for i in range(0, self.population_size):
                if np.random.rand() < self.cross_over_rate:
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                    mutant = np.clip(a + self.f * (b - c), lb, ub)
                    cross_points = np.random.rand(self.dim) < self.cross_over_rate
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                    population[i] = trial

            # Mutation
            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_rate
            mutation_values = np.random.uniform(lb, ub, (self.population_size, self.dim))
            population = np.where(mutation_mask, mutation_values, population)

        return global_best, global_best_score