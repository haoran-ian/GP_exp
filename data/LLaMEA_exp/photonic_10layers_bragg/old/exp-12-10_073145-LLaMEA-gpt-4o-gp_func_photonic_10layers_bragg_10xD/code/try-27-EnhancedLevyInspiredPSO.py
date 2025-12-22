import numpy as np

class EnhancedLevyInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = 50
        max_c1, max_c2 = 2.5, 2.5
        min_c1, min_c2 = 0.5, 0.5
        max_w, min_w = 1.0, 0.4

        # Initialize particles' positions and velocities
        positions = np.random.rand(num_particles, self.dim) * (ub - lb) + lb
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, np.inf)

        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0

        def levy_flight(Lambda):
            sigma1 = np.power((np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) /
                              (np.math.gamma((1 + Lambda) / 2) * Lambda * np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
            sigma2 = 1
            u = np.random.normal(0, sigma1, size=self.dim)
            v = np.random.normal(0, sigma2, size=self.dim)
            step = u / np.power(np.abs(v), 1 / Lambda)
            return 0.01 * step

        while evaluations < self.budget:
            for i in range(num_particles):
                score = func(positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            # Calculate diversity and adapt inertia weight with non-linear reduction
            diversity = np.mean(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1))
            w = max_w - (max_w - min_w) * ((evaluations / self.budget) ** 0.5)

            # Dynamic adjustment of cognitive and social coefficients
            c1 = min_c1 + (max_c1 - min_c1) * np.exp(-diversity / (ub - lb).mean())
            c2 = max_c2 - (max_c2 - min_c2) * np.exp(-diversity / (ub - lb).mean())

            # Lévy flight perturbation for exploration
            r1, r2 = np.random.rand(num_particles, self.dim), np.random.rand(num_particles, self.dim)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (global_best_position - positions))

            levy_step = levy_flight(1.5)
            positions += velocities + levy_step
            positions = np.clip(positions, lb, ub)

        return global_best_position, global_best_score