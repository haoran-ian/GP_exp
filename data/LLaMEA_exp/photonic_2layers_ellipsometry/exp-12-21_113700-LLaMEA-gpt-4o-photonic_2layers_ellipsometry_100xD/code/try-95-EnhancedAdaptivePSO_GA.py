import numpy as np

class EnhancedAdaptivePSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1_initial = 2.5  # initial cognitive parameter
        self.c2_initial = 0.5  # initial social parameter
        self.w_initial = 0.9   # initial inertia weight
        self.w_final = 0.4     # final inertia weight
        self.mutation_rate_initial = 0.2
        self.mutation_rate_final = 0.05
        self.cross_over_rate = 0.7
        self.elite_fraction = 0.1  # fraction of elite individuals

    def levy_flight(self, dim):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / np.abs(v)**(1 / beta)
        return step

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
            # Dynamic parameter adaptation
            w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)
            mutation_rate = self.mutation_rate_initial - (self.mutation_rate_initial - self.mutation_rate_final) * (evaluations / self.budget)

            # PSO Step with Levy Flight
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities + 
                          self.c1_initial * r1 * (personal_best - population) + 
                          self.c2_initial * r2 * (global_best - population))
            population = population + velocities + self.levy_flight(self.dim)
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

            # Elite selection
            elite_count = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(scores)[:elite_count]
            elite_population = population[elite_indices]

            # GA Step: Crossover
            for i in range(0, elite_count, 2):
                if np.random.rand() < self.cross_over_rate:
                    cross_point = np.random.randint(1, self.dim)
                    parent1, parent2 = elite_population[i], elite_population[min(i + 1, elite_count - 1)]
                    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                    elite_population[i], elite_population[i + 1] = child1, child2

            # Mutation
            mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate
            mutation_values = np.random.uniform(lb, ub, (self.population_size, self.dim))
            population = np.where(mutation_mask, mutation_values, population)

            # Reintegration of elite individuals
            population[:elite_count] = elite_population

            t += 1

        return global_best, global_best_score