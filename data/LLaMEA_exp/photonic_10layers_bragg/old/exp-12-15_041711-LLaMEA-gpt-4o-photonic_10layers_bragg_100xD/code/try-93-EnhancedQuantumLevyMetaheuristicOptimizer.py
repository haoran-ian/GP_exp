import numpy as np

class EnhancedQuantumLevyMetaheuristicOptimizer:
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

            chaotic_inertia = 0.9 - 0.5 * np.abs(np.sin(2 * np.pi * eval_count / self.budget))  # Modified chaotic inertia
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities = (chaotic_inertia * velocities +
                          cognitive_component * r1 * (personal_best_positions - particles) +
                          social_component * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, lb, ub)
            
            # Quantum-inspired superposition operator with Levy flight
            quantum_superposition = 0.5 * (personal_best_positions + global_best_position)
            adaptive_learning_rate = 0.1 + 0.9 * eval_count / self.budget
            levy_step = np.random.normal(size=(num_particles, self.dim)) * (1.5 / ((1.0 + np.abs(np.random.normal()))**(1 / 1.5)))
            superposed_particles = particles + adaptive_learning_rate * (quantum_superposition - particles) + levy_step
            superposed_particles = np.clip(superposed_particles, lb, ub)

            for i in range(num_particles):
                if eval_count >= self.budget:
                    break
                indices = list(range(num_particles))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                adaptive_differential_weight = differential_weight * np.sin(eval_count / self.budget)
                mutant_vector = personal_best_positions[a] + adaptive_differential_weight * (personal_best_positions[b] - personal_best_positions[c])
                chaos = np.sin(0.5 * eval_count * np.pi / self.budget)
                mutant_vector += chaos * (np.random.rand(self.dim) - 0.5) * 0.8
                cross_points = np.random.rand(self.dim) < (crossover_rate + 0.2 * chaos)
                trial_vector = np.where(cross_points, mutant_vector, superposed_particles[i])
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