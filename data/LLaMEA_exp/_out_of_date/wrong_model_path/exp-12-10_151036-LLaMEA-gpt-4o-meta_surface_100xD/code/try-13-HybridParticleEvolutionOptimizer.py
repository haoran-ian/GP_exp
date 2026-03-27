import numpy as np

class HybridParticleEvolutionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.temperature = 100
        self.cooling_rate = 0.95
        self.diversity_threshold = 0.1
        self.compression_factor = 0.5
        self.f = 0.5  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability
        
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
                # Differential Evolution Mutation and Crossover
                indices = list(range(self.num_particles))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(personal_best[a] + self.f * (personal_best[b] - personal_best[c]), lb, ub)
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, particles[i])

                # Evaluate Trial Individual
                trial_value = func(trial)
                eval_count += 1

                # Update Personal Best
                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < best_value:
                        global_best = trial
                        best_value = trial_value

                # Update Velocity and Position using Particle Swarm
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best[i] - particles[i]) +
                                 self.social_coeff * r2 * (global_best - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                
                if eval_count >= self.budget:
                    break
            
            # Adaptive inertia weight reduction
            self.inertia_weight = max(self.inertia_weight_min, self.inertia_weight * self.cooling_rate)
            
            # Simulated annealing-inspired acceptance mechanism
            if np.random.rand() < np.exp(-abs(trial_value - best_value) / self.temperature):
                global_best = trial
                best_value = trial_value
            
            self.temperature *= self.cooling_rate
            
            # Enhanced diversity preservation with compression
            if np.std(personal_best_values) < self.diversity_threshold:
                compressed_particles = particles + self.compression_factor * np.random.randn(*particles.shape)
                compressed_particles = np.clip(compressed_particles, lb, ub)
                compressed_values = np.array([func(x) for x in compressed_particles])
                eval_count += len(compressed_particles)
                
                for j in range(len(compressed_particles)):
                    if compressed_values[j] < best_value or np.random.rand() < 0.1:
                        global_best = compressed_particles[j]
                        best_value = compressed_values[j]
                        break

        return global_best