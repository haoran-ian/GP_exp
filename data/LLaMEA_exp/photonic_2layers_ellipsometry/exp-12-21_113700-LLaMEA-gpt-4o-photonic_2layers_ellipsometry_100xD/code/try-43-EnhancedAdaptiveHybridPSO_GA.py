import numpy as np

class EnhancedAdaptiveHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.sub_population_size = 10
        self.num_sub_populations = self.population_size // self.sub_population_size
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.w_initial = 0.9
        self.w_final = 0.4
        self.mutation_rate_initial = 0.2
        self.mutation_rate_final = 0.05
        self.cross_over_rate = 0.7
        self.elite_preservation_rate = 0.1

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
            # Linearly adapt inertia weight and mutation rate
            w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)
            mutation_rate = self.mutation_rate_initial - (self.mutation_rate_initial - self.mutation_rate_final) * (evaluations / self.budget)

            # Divide population into sub-populations
            sub_populations = np.array_split(population, self.num_sub_populations)
            sub_velocities = np.array_split(velocities, self.num_sub_populations)
            sub_personal_best = np.array_split(personal_best, self.num_sub_populations)
            sub_personal_best_scores = np.array_split(personal_best_scores, self.num_sub_populations)

            for i in range(self.num_sub_populations):
                sub_pop = sub_populations[i]
                sub_vel = sub_velocities[i]
                sub_pbest = sub_personal_best[i]
                sub_pbest_scores = sub_personal_best_scores[i]

                # Update velocities and positions (PSO Step)
                r1, r2 = np.random.rand(self.sub_population_size, self.dim), np.random.rand(self.sub_population_size, self.dim)
                sub_vel = (w * sub_vel + 
                           self.c1_initial * r1 * (sub_pbest - sub_pop) + 
                           self.c2_initial * r2 * (global_best - sub_pop))
                sub_pop += sub_vel
                sub_pop = np.clip(sub_pop, lb, ub)

                # Evaluate sub-population
                sub_scores = np.array([func(ind) for ind in sub_pop])
                evaluations += self.sub_population_size

                # Update personal best and global best
                improved = sub_scores < sub_pbest_scores
                sub_pbest[improved] = sub_pop[improved]
                sub_pbest_scores[improved] = sub_scores[improved]
                if min(sub_scores) < global_best_score:
                    global_best = sub_pop[np.argmin(sub_scores)]
                    global_best_score = min(sub_scores)

                # Apply Genetic Operators (GA Step)
                # Crossover
                for j in range(0, self.sub_population_size, 2):
                    if np.random.rand() < self.cross_over_rate:
                        cross_point = np.random.randint(1, self.dim)
                        parent1, parent2 = sub_pop[j], sub_pop[j + 1]
                        child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                        child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                        sub_pop[j], sub_pop[j + 1] = child1, child2

                # Mutation
                mutation_mask = np.random.rand(self.sub_population_size, self.dim) < mutation_rate
                mutation_values = np.random.uniform(lb, ub, (self.sub_population_size, self.dim))
                sub_pop = np.where(mutation_mask, mutation_values, sub_pop)

                # Update sub-population
                sub_populations[i] = sub_pop
                sub_velocities[i] = sub_vel
                sub_personal_best[i] = sub_pbest
                sub_personal_best_scores[i] = sub_pbest_scores

            # Merge sub-populations
            population = np.vstack(sub_populations)
            velocities = np.vstack(sub_velocities)
            personal_best = np.vstack(sub_personal_best)
            personal_best_scores = np.concatenate(sub_personal_best_scores)

            # Elite Preservation
            elite_count = int(self.population_size * self.elite_preservation_rate)
            elite_indices = np.argsort(personal_best_scores)[:elite_count]
            elite_population = personal_best[elite_indices]
            population[:elite_count] = elite_population

        return global_best, global_best_score