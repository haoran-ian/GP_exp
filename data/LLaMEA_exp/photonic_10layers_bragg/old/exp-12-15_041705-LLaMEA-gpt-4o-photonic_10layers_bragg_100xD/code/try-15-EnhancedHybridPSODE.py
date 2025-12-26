import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20  # Initial Population size
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.w = 0.5  # Inertia weight
        self.cr = 0.9  # Crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_pop_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (pop_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.full(pop_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        eval_count = 0

        while eval_count < self.budget:
            # Adjust population size dynamically
            pop_size_dynamic = self.initial_pop_size - int((self.initial_pop_size / 2) * (eval_count / self.budget))
            pop = pop[:pop_size_dynamic]
            velocities = velocities[:pop_size_dynamic]
            personal_best_positions = personal_best_positions[:pop_size_dynamic]
            personal_best_scores = personal_best_scores[:pop_size_dynamic]

            scores = np.apply_along_axis(func, 1, pop)
            eval_count += pop_size_dynamic

            # Update personal bests and global best
            for i in range(pop_size_dynamic):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = pop[i]

                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = pop[i]

            # PSO update with dynamic inertia weight
            r1, r2 = np.random.rand(pop_size_dynamic, self.dim), np.random.rand(pop_size_dynamic, self.dim)
            w_dynamic = 0.4 + (0.6 - 0.4) * (self.budget - eval_count) / self.budget
            velocities = (w_dynamic * velocities +
                          self.c1 * r1 * (personal_best_positions - pop) +
                          self.c2 * r2 * (global_best_position - pop))
            pop = pop + velocities

            # DE update
            for i in range(pop_size_dynamic):
                idxs = np.random.choice(np.delete(np.arange(pop_size_dynamic), i), 3, replace=False)
                x1, x2, x3 = pop[idxs]
                f_dynamic = 0.5 + 0.4 * (1 - (eval_count / self.budget))  # Enhanced adaptive differential weight
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