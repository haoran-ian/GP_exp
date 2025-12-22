import numpy as np

class QuantumAdaptiveMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        num_particles = 30
        inertia_weight = 0.9
        cognitive_component = 1.5
        social_component = 1.5
        differential_weight = 0.8
        crossover_rate = 0.9
        
        # Initialize particles
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([float('inf')] * num_particles)
        
        # Global best initialization
        global_best_position = None
        global_best_score = float('inf')
        
        eval_count = 0

        while eval_count < self.budget:
            for i in range(num_particles):
                score = func(particles[i])
                eval_count += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

                if eval_count >= self.budget:
                    break

            inertia_weight = 0.7 - (0.4 * eval_count / self.budget)  # Adjusted inertia weight
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities = (0.9 * inertia_weight * velocities +
                          1.3 * cognitive_component * r1 * (personal_best_positions - particles) +
                          0.9 * social_component * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, lb, ub)

            for i in range(num_particles):
                if eval_count >= self.budget:
                    break
                indices = list(range(num_particles))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                adaptive_differential_weight = differential_weight * np.sin(eval_count / self.budget) 
                quantum_perturbation = 0.1 * np.random.randn(self.dim)  # Quantum-inspired perturbation
                mutant_vector = personal_best_positions[a] + adaptive_differential_weight * (personal_best_positions[b] - personal_best_positions[c]) + quantum_perturbation
                chaos = np.sin(0.5 * eval_count * np.pi / self.budget)  
                mutant_vector += chaos * (np.random.rand(self.dim) - 0.5) * 0.8  
                cross_points = np.random.rand(self.dim) < (crossover_rate + 0.1 * np.sin(eval_count / self.budget))  
                trial_vector = np.where(cross_points, mutant_vector, particles[i])
                trial_vector = np.clip(trial_vector, lb, ub)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial_vector

        return global_best_position, global_best_score