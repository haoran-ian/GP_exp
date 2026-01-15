import numpy as np

class EPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        num_particles = 20
        w_max = 0.9
        w_min = 0.4
        c1_max = 2.5
        c1_min = 0.5
        c2_max = 2.5
        c2_min = 0.5

        positions = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf

        num_evaluations = 0
        success_history = np.zeros(num_particles)

        def update_parameters(success_rate, iter_fraction):
            w = w_min + (w_max - w_min) * (1 - success_rate) * (1 - iter_fraction)
            c1 = c1_min + (c1_max - c1_min) * (1 - success_rate)
            c2 = c2_min + (c2_max - c2_min) * success_rate
            return w, c1, c2

        while num_evaluations < self.budget:
            iter_fraction = num_evaluations / self.budget

            for i in range(num_particles):
                score = func(positions[i])
                num_evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                    success_history[i] += 1

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            diversity = np.mean(np.std(positions, axis=0))
            success_rate = np.mean(success_history) / (num_evaluations / num_particles)
            w, c1, c2 = update_parameters(success_rate, iter_fraction)

            for i in range(num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                positions[i] += velocities[i]

                positions[i] = np.clip(positions[i], lb, ub)

            success_history *= 0.9  # decay to gradually forget old success

        return global_best_position, global_best_score