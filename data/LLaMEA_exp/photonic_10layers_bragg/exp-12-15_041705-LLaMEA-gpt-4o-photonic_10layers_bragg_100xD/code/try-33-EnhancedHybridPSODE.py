import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20  # Initial population size
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.w = 0.5  # Inertia weight
        self.f = 0.8  # Differential weight
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
            scores = np.apply_along_axis(func, 1, pop)
            eval_count += pop_size

            # Update personal bests and global best
            for i in range(pop_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = pop[i]

                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = pop[i]

            # Update population size dynamically
            new_pop_size = self.initial_pop_size + int((self.budget - eval_count) / self.budget * 10)
            if new_pop_size != pop_size:
                additional_pop = np.random.uniform(lb, ub, (new_pop_size - pop_size, self.dim))
                pop = np.vstack((pop, additional_pop))
                velocities = np.vstack((velocities, np.random.uniform(-1, 1, (new_pop_size - pop_size, self.dim))))
                personal_best_positions = np.vstack((personal_best_positions, additional_pop))
                personal_best_scores = np.append(personal_best_scores, np.full(new_pop_size - pop_size, np.inf))
                pop_size = new_pop_size

            # PSO update with dynamic inertia weight
            r1, r2 = np.random.rand(pop_size, self.dim), np.random.rand(pop_size, self.dim)
            w_dynamic = 0.4 + (0.5 - 0.4) * (self.budget - eval_count) / self.budget
            velocities = (w_dynamic * velocities +
                          self.c1 * r1 * (personal_best_positions - pop) +
                          self.c2 * r2 * (global_best_position - pop))
            pop = pop + velocities

            # DE update with adaptive mutation
            for i in range(pop_size):
                idxs = np.random.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
                x1, x2, x3 = pop[idxs]
                f_dynamic = 0.5 + 0.3 * (eval_count / self.budget) * (1 if np.random.rand() > 0.5 else -1)
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