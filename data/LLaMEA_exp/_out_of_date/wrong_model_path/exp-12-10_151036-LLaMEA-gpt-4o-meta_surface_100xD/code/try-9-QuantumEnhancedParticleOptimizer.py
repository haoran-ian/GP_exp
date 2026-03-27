import numpy as np

class QuantumEnhancedParticleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.warmup = 0.3 * budget
        self.diversity_threshold = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(x) for x in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        best_value = min(personal_best_values)

        eval_count = self.num_particles

        while eval_count < self.budget:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best[i] - particles[i]) +
                                 self.social_coeff * r2 * (global_best - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                current_value = func(particles[i])
                eval_count += 1

                if current_value < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = current_value

                    if current_value < best_value:
                        global_best = particles[i]
                        best_value = current_value

                if eval_count >= self.budget:
                    break

            # Dynamic inertia weight adjustment
            self.inertia_weight = max(self.inertia_weight_min, 
                                      self.inertia_weight * (1 - eval_count / self.budget))
            
            # Quantum-inspired exploration
            if eval_count > self.warmup and np.std(personal_best_values) < self.diversity_threshold:
                Q = np.random.uniform(-1, 1, (self.num_particles, self.dim))
                quantum_particles = global_best + Q * (particles - global_best)
                quantum_particles = np.clip(quantum_particles, lb, ub)
                quantum_values = np.array([func(x) for x in quantum_particles])
                eval_count += len(quantum_particles)

                for j in range(len(quantum_particles)):
                    if quantum_values[j] < best_value:
                        global_best = quantum_particles[j]
                        best_value = quantum_values[j]
                        break

        return global_best