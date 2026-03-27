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
        c1_base = 2.0
        c2_base = 2.0

        positions = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf

        num_evaluations = 0

        def update_parameters(phase, diversity):
            if phase < 0.3:
                # Exploration phase
                c1 = c1_base + 1.0 * diversity
                c2 = c2_base - 0.5 * diversity
            elif phase < 0.7:
                # Transition phase
                c1 = c1_base
                c2 = c2_base
            else:
                # Exploitation phase
                c1 = c1_base - 0.5 * diversity
                c2 = c2_base + 1.0 * diversity
            w = w_max - (w_max - w_min) * phase
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

            diversity = np.mean(np.std(positions, axis=0))
            w, c1, c2 = update_parameters(iter_fraction, diversity)

            for i in range(num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                positions[i] += velocities[i]

                positions[i] = np.clip(positions[i], lb, ub)

        return global_best_position, global_best_score