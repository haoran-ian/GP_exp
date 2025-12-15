import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.c1_init = 2.5
        self.c2_init = 0.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.f = 0.8
        self.cr = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.full(self.pop_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        eval_count = 0

        while eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, pop)
            eval_count += self.pop_size

            for i in range(self.pop_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = pop[i]

                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = pop[i]

            w_dynamic = self.w_max - ((self.w_max - self.w_min) * (eval_count / self.budget))
            c1_dynamic = self.c1_init * (1 - eval_count / self.budget)
            c2_dynamic = self.c2_init + (2.5 - self.c2_init) * (eval_count / self.budget)

            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (w_dynamic * velocities +
                          c1_dynamic * r1 * (personal_best_positions - pop) +
                          c2_dynamic * r2 * (global_best_position - pop))
            pop = np.clip(pop + velocities, lb, ub)

            for i in range(self.pop_size):
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x1, x2, x3 = pop[idxs]
                f_dynamic = 0.5 + 0.3 * (eval_count / self.budget)
                mutant = np.clip(x1 + f_dynamic * (x2 - x3), lb, ub)
                cross_points = np.random.rand(self.dim) < (self.cr * (eval_count / self.budget))
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                trial_score = func(trial)
                eval_count += 1

                if trial_score < scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score

                # Local search with mutation
                if np.random.rand() < 0.1:
                    mutation = np.random.normal(0, 0.1, self.dim)
                    mutated_individual = np.clip(pop[i] + mutation, lb, ub)
                    mutated_score = func(mutated_individual)
                    eval_count += 1

                    if mutated_score < scores[i]:
                        pop[i] = mutated_individual
                        scores[i] = mutated_score

        return global_best_position, global_best_score