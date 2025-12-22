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

        # Dynamic memory for adaptive exploration
        memory_positions = np.copy(positions)
        memory_scores = np.full(num_particles, np.inf)

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
                    
                # Update memory positions
                if score < memory_scores[i]:
                    memory_scores[i] = score
                    memory_positions[i] = positions[i]

            diversity = np.mean(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1))
            w = max_w - (max_w - min_w) * ((evaluations / self.budget) ** (1.0 / 3))

            c1 = min_c1 + (max_c1 - min_c1) * np.exp(-diversity / (ub - lb).mean())
            c2 = max_c2 - (max_c2 - min_c2) * np.exp(-diversity / (ub - lb).mean())

            perturbation_factor = np.random.normal(0, 0.1, (num_particles, self.dim))
            r1, r2 = np.random.rand(num_particles, self.dim), np.random.rand(num_particles, self.dim)

            convergence_rate = (self.budget - evaluations) / self.budget
            quantum_factor = np.random.uniform(0.5, 1.5 * convergence_rate, (num_particles, self.dim))
            
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (global_best_position - positions))
            
            # Adaptive perturbation based on diversity
            adaptive_perturbation = perturbation_factor * (1 + diversity / (ub - lb).mean())
            perturbation = adaptive_perturbation * np.sign(velocities) * np.abs(positions - global_best_position)
            positions += quantum_factor * perturbation
            positions = np.clip(positions, lb, ub)
            
            # Lévy flight perturbation with dynamic memory
            levy = np.random.standard_cauchy((num_particles, self.dim))
            positions += levy * 0.01 * (memory_positions - positions) + adaptive_perturbation * 0.01

        return global_best_position, global_best_score