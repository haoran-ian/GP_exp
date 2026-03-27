import numpy as np

class EAPSO:
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
        prev_global_best_score = np.inf

        def update_parameters(diversity, iter_fraction, convergence_rate):
            w = w_max - (w_max - w_min) * iter_fraction
            c1 = (c1_max - c1_min) * (1 - convergence_rate) + c1_min
            c2 = (c2_max - c2_min) * convergence_rate + c2_min
            return w, c1, c2

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

            convergence_rate = np.abs(global_best_score - prev_global_best_score) / (np.abs(prev_global_best_score) + 1e-10)
            prev_global_best_score = global_best_score

            diversity = np.mean(np.std(positions, axis=0))
            w, c1, c2 = update_parameters(diversity, iter_fraction, convergence_rate)

            for i in range(num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                positions[i] += velocities[i]

                positions[i] = np.clip(positions[i], lb, ub)

        return global_best_position, global_best_score