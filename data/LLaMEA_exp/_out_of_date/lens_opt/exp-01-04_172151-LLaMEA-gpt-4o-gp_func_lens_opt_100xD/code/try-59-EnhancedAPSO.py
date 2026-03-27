import numpy as np

class EnhancedAPSO:
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

        def update_parameters(diversity, iter_fraction):
            w = w_max - (w_max - w_min) * iter_fraction
            c1 = c1_max - (c1_max - c1_min) * diversity
            c2 = c2_min + (c2_max - c2_min) * diversity
            return w, c1, c2
        
        def levy_flight(Lambda):
            sigma1 = np.power((np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) / 
                              (np.math.gamma((1 + Lambda) / 2) * Lambda * 
                               np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
            u = np.random.normal(0, sigma1, size=self.dim)
            v = np.random.normal(0, 1, size=self.dim)
            step = u / np.power(np.abs(v), 1 / Lambda)
            L = 0.01 * step
            return L

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
            w, c1, c2 = update_parameters(diversity, iter_fraction)

            for i in range(num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social

                # Implementing velocity clamping
                velocity_max = (ub - lb) / 10.0
                velocities[i] = np.clip(velocities[i], -velocity_max, velocity_max)

                # Integrating Lévy Flight for exploration
                if np.random.rand() < 0.01:
                    step = levy_flight(1.5)
                    positions[i] = positions[i] + step * (positions[i] - global_best_position)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

        return global_best_position, global_best_score