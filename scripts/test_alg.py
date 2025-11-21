import numpy as np


class DE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, self.budget // 10)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        np.random.seed(0)
        pop = self.lower_bound + (self.upper_bound - self.lower_bound) * \
            np.random.rand(self.population_size, self.dim)
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evals = self.population_size

        while evals < self.budget:
            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                a, b, c = np.random.choice(
                    np.delete(np.arange(self.population_size), i), 3, replace=False)
                scaled_mutation_factor = self.mutation_factor * \
                    (1 - (global_best_score /
                     personal_best_scores[i]))  # Changed line
                mutant = np.clip(pop[a] + scaled_mutation_factor * (pop[b] - pop[c]),
                                 self.lower_bound, self.upper_bound)  # Changed line
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_score = func(trial)
                evals += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best[i] = trial

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best = trial

            # Particle Swarm Optimization Update
            w = self.w_max - (self.w_max - self.w_min) * \
                (evals / self.budget)**2  # Changed line
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(
                self.population_size, self.dim)
            velocities = w * velocities + self.c1 * r1 * \
                (personal_best - pop) + self.c2 * r2 * (global_best - pop)
            pop = np.clip(pop + velocities, self.lower_bound, self.upper_bound)
        print(global_best)
        return global_best
