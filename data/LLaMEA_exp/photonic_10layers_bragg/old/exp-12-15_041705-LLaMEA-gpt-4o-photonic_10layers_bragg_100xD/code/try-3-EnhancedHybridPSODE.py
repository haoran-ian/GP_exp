import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.c1_initial, self.c2_initial = 1.5, 1.5
        self.w_initial, self.w_final = 0.9, 0.4
        self.f = 0.8
        self.cr = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)  # Ensure reproducibility
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim) * (np.random.rand(self.pop_size, self.dim) > 0.5)
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.full(self.pop_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        eval_count = 0

        while eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, pop)
            eval_count += self.pop_size

            # Update personal bests and global best
            for i in range(self.pop_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = pop[i]

                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = pop[i]

            # Adaptive PSO parameters
            w = self.w_final + (self.w_initial - self.w_final) * ((self.budget - eval_count) / self.budget)
            c1 = self.c1_initial * (eval_count / self.budget)
            c2 = self.c2_initial * (eval_count / self.budget)

            # PSO update
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - pop) +
                          c2 * r2 * (global_best_position - pop))
            pop = np.clip(pop + velocities, lb, ub)

            # DE update
            for i in range(self.pop_size):
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x1, x2, x3 = pop[idxs]
                mutant = np.clip(x1 + self.f * (x2 - x3), lb, ub)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                trial_score = func(trial)
                eval_count += 1

                if trial_score < scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score

        return global_best_position, global_best_score