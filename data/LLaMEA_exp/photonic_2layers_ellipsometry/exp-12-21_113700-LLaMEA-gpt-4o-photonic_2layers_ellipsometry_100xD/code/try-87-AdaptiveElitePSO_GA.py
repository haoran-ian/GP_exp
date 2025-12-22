import numpy as np

class AdaptiveElitePSO_GA:
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
        self.elite_fraction = 0.2  # Proportion of elite individuals to be retained

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
            # Linearly adapt inertia weight and mutation rate
            w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)
            mutation_rate = self.mutation_rate_initial - (self.mutation_rate_initial - self.mutation_rate_final) * (evaluations / self.budget)

            # Update velocities and population positions (PSO Step)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities + 
                          self.c1_initial * r1 * (personal_best - population) + 
                          self.c2_initial * r2 * (global_best - population))
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
            
            # Elite selection
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(scores)[:elite_size]
            elite_population = population[elite_indices]
            elite_scores = scores[elite_indices]

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
            mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate
            mutation_values = np.random.uniform(lb, ub, (self.population_size, self.dim))
            population = np.where(mutation_mask, mutation_values, population)

            # Reintroduce elite individuals to maintain diversity
            population[:elite_size] = elite_population
            personal_best_scores[:elite_size] = elite_scores

            t += 1

        return global_best, global_best_score