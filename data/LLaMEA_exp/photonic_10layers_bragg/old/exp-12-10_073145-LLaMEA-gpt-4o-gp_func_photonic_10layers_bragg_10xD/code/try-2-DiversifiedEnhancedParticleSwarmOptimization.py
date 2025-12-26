import numpy as np

class DiversifiedEnhancedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = 50
        max_c1, max_c2 = 2.5, 2.5
        min_c1, min_c2 = 0.5, 0.5
        max_w, min_w = 0.9, 0.4
        reinit_prob = 0.1  # Probability of reinitialization for stagnant particles

        # Initialize particles' positions and velocities
        positions = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, np.inf)

        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0
        stagnation_counts = np.zeros(num_particles)

        while evaluations < self.budget:
            for i in range(num_particles):
                score = func(positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                    stagnation_counts[i] = 0  # Reset stagnation counter
                else:
                    stagnation_counts[i] += 1

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            # Calculate diversity and adapt inertia weight
            diversity = np.mean(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1))
            w = max_w - (max_w - min_w) * (evaluations / self.budget)
            
            # Dynamic adjustment of cognitive and social coefficients
            c1 = min_c1 + (max_c1 - min_c1) * (1 - diversity / (ub - lb).mean())
            c2 = max_c2 - (max_c2 - min_c2) * (1 - diversity / (ub - lb).mean())

            # Reinitialize stagnated particles with a probability
            for i in range(num_particles):
                if np.random.rand() < reinit_prob * stagnation_counts[i] / self.budget:
                    positions[i] = np.random.uniform(lb, ub, self.dim)
                    velocities[i] = np.random.uniform(-1, 1, self.dim)
                    stagnation_counts[i] = 0

            # Update velocities and positions
            r1, r2 = np.random.rand(num_particles, self.dim), np.random.rand(num_particles, self.dim)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, lb, ub)

        return global_best_position, global_best_score