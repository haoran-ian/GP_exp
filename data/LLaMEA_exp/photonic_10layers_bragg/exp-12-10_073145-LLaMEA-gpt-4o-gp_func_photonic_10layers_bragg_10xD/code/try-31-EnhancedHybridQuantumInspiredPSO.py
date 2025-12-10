import numpy as np

class EnhancedHybridQuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = 50
        max_c1, max_c2 = 2.5, 2.5
        min_c1, min_c2 = 0.5, 0.5
        max_w, min_w = 1.0, 0.4

        # Initialize particles' positions and velocities with Gaussian distribution
        positions = np.random.normal((lb + ub) / 2, (ub - lb) / 4, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, np.inf)

        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0

        def levy_flight(Lambda):
            # Levy flight implementation
            sigma_u = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
                       (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
            u = np.random.normal(0, sigma_u, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / abs(v) ** (1 / Lambda)
            return step

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

            # Gaussian-driven perturbation
            perturbation_factor = np.random.normal(0, 0.1, (num_particles, self.dim))
            r1, r2 = np.random.rand(num_particles, self.dim), np.random.rand(num_particles, self.dim)
            quantum_factor = np.random.uniform(0.5, 1.5, (num_particles, self.dim))
            
            # Self-adaptive learning and levy flight for exploration
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (global_best_position - positions))
            
            perturbation = perturbation_factor * np.sign(velocities) * np.abs(positions - global_best_position)
            levy_steps = np.array([levy_flight(1.5) for _ in range(num_particles)])
            
            exploration_update = quantum_factor * perturbation + levy_steps
            positions += exploration_update
            positions = np.clip(positions, lb, ub)

        return global_best_position, global_best_score