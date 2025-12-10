import numpy as np

class EnhancedQuantumInspiredParticleSwarmOptimization:
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

            diversity = np.mean(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1))
            w = max_w - (max_w - min_w) * ((global_best_score / np.mean(personal_best_scores)) ** 2)

            c1 = min_c1 + (max_c1 - min_c1) * np.exp(-diversity / (ub - lb).mean())
            c2 = max_c2 - (max_c2 - min_c2) * np.exp(-diversity / (ub - lb).mean())

            perturbation_factor = np.random.normal(0, 0.1, (num_particles, self.dim))
            r1, r2 = np.random.rand(num_particles, self.dim), np.random.rand(num_particles, self.dim)

            # Dynamic neighborhood-based update
            neighborhood_size = int(num_particles * 0.1)
            for i in range(num_particles):
                neighbors = np.random.choice(num_particles, neighborhood_size, replace=False)
                local_best_index = neighbors[np.argmin(personal_best_scores[neighbors])]
                local_best_position = personal_best_positions[local_best_index]

                velocities[i] = (w * velocities[i] +
                                 c1 * r1[i] * (personal_best_positions[i] - positions[i]) +
                                 c2 * r2[i] * (local_best_position - positions[i]))

            perturbation = perturbation_factor * np.sign(velocities) * np.abs(positions - global_best_position)
            positions += velocities + perturbation
            positions = np.clip(positions, lb, ub)

            # Lévy flight perturbation
            levy = np.random.standard_cauchy((num_particles, self.dim))
            positions += levy * perturbation_factor * 0.01
            
        return global_best_position, global_best_score