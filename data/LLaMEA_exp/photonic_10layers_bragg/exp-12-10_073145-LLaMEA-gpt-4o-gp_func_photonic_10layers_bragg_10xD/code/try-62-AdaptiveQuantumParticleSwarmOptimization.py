import numpy as np

class AdaptiveQuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = 50
        max_c1, max_c2 = 2.5, 2.5
        min_c1, min_c2 = 0.5, 0.5
        max_w, min_w = 1.0, 0.4

        positions = np.random.normal((lb + ub) / 2, (ub - lb) / 4, (num_particles, self.dim))
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

            diversity = np.std(positions, axis=0).mean()
            w = max_w - (max_w - min_w) * (evaluations / self.budget)

            c1 = min_c1 + (max_c1 - min_c1) * np.exp(-diversity / (ub - lb).mean())
            c2 = max_c2 - (max_c2 - min_c2) * np.exp(-diversity / (ub - lb).mean())

            r1, r2 = np.random.rand(num_particles, self.dim), np.random.rand(num_particles, self.dim)

            # Adaptive quantum factor and enhanced dynamic swarm intelligence
            quantum_factor = np.random.uniform(0.5, 1.5, (num_particles, self.dim))
            adaptive_momentum = 0.9 + 0.1 * (1 - evaluations / self.budget)

            velocities = adaptive_momentum * (w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (global_best_position - positions))

            new_positions = positions + velocities * quantum_factor
            positions = np.clip(new_positions, lb, ub)

            # Enhanced Lévy flight with self-adaptive resilience factor
            levy_step = np.random.standard_cauchy((num_particles, self.dim))
            adaptive_resilience = 1.0 + 0.1 * np.sin(evaluations / self.budget * np.pi)
            positions += adaptive_resilience * levy_step * quantum_factor * 0.01

        return global_best_position, global_best_score