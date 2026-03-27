import numpy as np

class EnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        num_particles = 20
        w_init = 0.9
        w_end = 0.4
        c1 = 2.05
        c2 = 2.05
        velocity_clamp = 0.5 * (ub - lb)

        positions = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf

        num_evaluations = 0

        def update_inertia_weight(iter_fraction):
            return w_init - (w_init - w_end) * iter_fraction

        while num_evaluations < self.budget:
            iter_fraction = num_evaluations / self.budget

            for i in range(num_particles):
                score = func(positions[i])
                num_evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            w = update_inertia_weight(iter_fraction)

            for i in range(num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social

                # Apply velocity clamping
                velocities[i] = np.clip(velocities[i], -velocity_clamp, velocity_clamp)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

        return global_best_position, global_best_score