import numpy as np

class AdaptiveQuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = 50
        max_c1, max_c2 = 2.5, 2.5
        min_c1, min_c2 = 0.5, 0.5
        max_w, min_w = 0.9, 0.4

        positions = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

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

            diversity = np.mean(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1))
            w = max_w - (max_w - min_w) * ((evaluations / self.budget) ** (1.0 / 2))

            c1 = min_c1 + (max_c1 - min_c1) * np.exp(-diversity / (ub - lb).mean())
            c2 = max_c2 - (max_c2 - min_c2) * np.exp(-diversity / (ub - lb).mean())

            perturbation_factor = np.random.normal(0, 0.1, (num_particles, self.dim))
            r1, r2 = np.random.rand(num_particles, self.dim), np.random.rand(num_particles, self.dim)

            # Improved dynamic quantum factor with adaptive momentum
            convergence_progress = evaluations / self.budget
            quantum_factor = np.random.uniform(0.5, 1.5 * (1 - convergence_progress), (num_particles, self.dim))
            momentum_factor = 0.8 + 0.2 * (1 - convergence_progress)

            velocities = momentum_factor * (w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (global_best_position - positions))

            chaotic_factor = np.sin(np.pi * convergence_progress)
            positions += quantum_factor * velocities + chaotic_factor * perturbation_factor
            positions = np.clip(positions, lb, ub)

            # Lévy flight perturbation with resilience factor
            levy = np.random.standard_cauchy((num_particles, self.dim))
            resilience_factor = np.random.uniform(0.95, 1.05)
            positions += resilience_factor * levy * perturbation_factor * 0.01

        return global_best_position, global_best_score