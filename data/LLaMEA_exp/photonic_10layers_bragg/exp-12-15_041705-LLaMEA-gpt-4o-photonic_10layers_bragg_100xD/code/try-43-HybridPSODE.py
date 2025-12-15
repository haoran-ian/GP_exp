import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
        self.c1_init = 2.5  # Initial cognitive coefficient
        self.c2_init = 0.5  # Initial social coefficient
        self.w = 0.5  # Inertia weight
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability

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

            # Update personal bests and global best
            for i in range(self.pop_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = pop[i]

                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = pop[i]

            # Adaptive learning rates for cognitive and social components
            c1_dynamic = self.c1_init - (self.c1_init - self.c2_init) * (eval_count / self.budget)
            c2_dynamic = self.c2_init + (self.c1_init - self.c2_init) * (eval_count / self.budget)

            # PSO update with dynamic inertia weight
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            w_dynamic = 0.4 + (0.5 - 0.4) * np.exp(-5 * eval_count / self.budget)
            velocities = (w_dynamic * velocities +
                          c1_dynamic * r1 * (personal_best_positions - pop) +
                          c2_dynamic * r2 * (global_best_position - pop))
            pop = pop + velocities

            # DE update
            for i in range(self.pop_size):
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x1, x2, x3 = pop[idxs]
                f_dynamic = 0.5 + 0.5 * np.sin(np.pi * eval_count / self.budget)  # Modified line
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

        return global_best_position, global_best_score