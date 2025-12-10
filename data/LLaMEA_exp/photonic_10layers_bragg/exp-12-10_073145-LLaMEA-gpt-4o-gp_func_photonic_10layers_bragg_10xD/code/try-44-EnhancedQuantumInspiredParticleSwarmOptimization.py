import numpy as np

class EnhancedQuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = 50
        num_swarms = 3
        swarm_size = num_particles // num_swarms
        max_c1, max_c2 = 2.5, 2.5
        min_c1, min_c2 = 0.5, 0.5
        max_w, min_w = 1.0, 0.4

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
            w = max_w - (max_w - min_w) * (evaluations / self.budget)

            c1 = min_c1 + (max_c1 - min_c1) * np.exp(-diversity / (ub - lb).mean())
            c2 = max_c2 - (max_c2 - min_c2) * np.exp(-diversity / (ub - lb).mean())

            perturbation_factor = np.random.normal(0, 0.1, (num_particles, self.dim))
            r1, r2 = np.random.rand(num_particles, self.dim), np.random.rand(num_particles, self.dim)

            # Multi-swarm strategy
            for s in range(num_swarms):
                swarm_slice = slice(s * swarm_size, (s + 1) * swarm_size)
                velocities[swarm_slice] = (
                    w * velocities[swarm_slice] +
                    c1 * r1[swarm_slice] * (personal_best_positions[swarm_slice] - positions[swarm_slice]) +
                    c2 * r2[swarm_slice] * (global_best_position - positions[swarm_slice])
                )
                perturbation = perturbation_factor[swarm_slice] * np.sign(velocities[swarm_slice]) * np.abs(positions[swarm_slice] - global_best_position)
                quantum_factor = np.random.uniform(0.5, 1.5, (swarm_size, self.dim))
                positions[swarm_slice] += quantum_factor * perturbation
                positions[swarm_slice] = np.clip(positions[swarm_slice], lb, ub)

            # Lévy flight perturbation
            levy = np.random.standard_cauchy((num_particles, self.dim))
            positions += levy * perturbation_factor * 0.01
            
        return global_best_position, global_best_score