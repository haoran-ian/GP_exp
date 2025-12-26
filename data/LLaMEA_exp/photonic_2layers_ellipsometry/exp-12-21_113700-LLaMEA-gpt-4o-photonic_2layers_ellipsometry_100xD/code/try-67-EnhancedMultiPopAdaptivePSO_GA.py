import numpy as np

class EnhancedMultiPopAdaptivePSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.sub_populations = 3
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.w_initial = 0.9
        self.w_final = 0.4
        self.mutation_rate_initial = 0.2
        self.mutation_rate_final = 0.05
        self.cross_over_rate = 0.7
        self.learning_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        sub_pop_size = self.population_size // self.sub_populations
        populations = [np.random.uniform(lb, ub, (sub_pop_size, self.dim)) for _ in range(self.sub_populations)]
        velocities = [np.random.uniform(-1, 1, (sub_pop_size, self.dim)) for _ in range(self.sub_populations)]
        personal_bests = [pop.copy() for pop in populations]
        personal_best_scores = [np.array([func(ind) for ind in pop]) for pop in populations]
        global_best = np.vstack(personal_bests)[np.argmin([min(scores) for scores in personal_best_scores])]
        global_best_score = min([min(scores) for scores in personal_best_scores])

        evaluations = sub_pop_size * self.sub_populations

        while evaluations < self.budget:
            w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)
            mutation_rate = self.mutation_rate_initial - (self.mutation_rate_initial - self.mutation_rate_final) * (evaluations / self.budget)

            for i, (population, velocity, personal_best, personal_best_score) in enumerate(zip(populations, velocities, personal_bests, personal_best_scores)):
                r1, r2 = np.random.rand(sub_pop_size, self.dim), np.random.rand(sub_pop_size, self.dim)
                velocities[i] = (w * velocity +
                                 self.c1_initial * r1 * (personal_best - population) +
                                 self.c2_initial * r2 * (global_best - population))
                populations[i] = population + velocities[i]
                populations[i] = np.clip(populations[i], lb, ub)

                scores = np.array([func(ind) for ind in populations[i]])
                evaluations += sub_pop_size

                improved = scores < personal_best_score
                personal_best[improved] = populations[i][improved]
                personal_best_score[improved] = scores[improved]

                if min(scores) < global_best_score:
                    global_best = populations[i][np.argmin(scores)]
                    global_best_score = min(scores)

                # Adaptive learning rate update
                self.c1_initial += self.learning_rate * np.mean(scores - personal_best_score)
                self.c2_initial -= self.learning_rate * np.mean(scores - global_best_score)

            # Apply Genetic Operators
            for population in populations:
                for i in range(0, sub_pop_size, 2):
                    if np.random.rand() < self.cross_over_rate:
                        cross_point = np.random.randint(1, self.dim)
                        parent1, parent2 = population[i], population[i + 1]
                        child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                        child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                        population[i], population[i + 1] = child1, child2

                mutation_mask = np.random.rand(sub_pop_size, self.dim) < mutation_rate
                mutation_values = np.random.uniform(lb, ub, (sub_pop_size, self.dim))
                population[:] = np.where(mutation_mask, mutation_values, population)

        return global_best, global_best_score