import numpy as np

class MultiStrategyParticleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.diversity_threshold = 0.1
        self.alpha = 0.1
        self.epsilon = 1e-6
        self.velocity_clamp = 0.2
        self.elite_archive_size = 5
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(x) for x in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        best_value = min(personal_best_values)
        elite_archive = [global_best]
        
        eval_count = self.num_particles
        
        while eval_count < self.budget:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = np.clip(
                    self.inertia_weight * velocities[i] +
                    self.cognitive_coeff * r1 * (personal_best[i] - particles[i]) +
                    self.social_coeff * r2 * (global_best - particles[i]),
                    -self.velocity_clamp, self.velocity_clamp
                )
                
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
            
            # Update elite archive
            if current_value < min(elite_archive, key=func):
                elite_archive.append(global_best)
                if len(elite_archive) > self.elite_archive_size:
                    elite_archive.pop(0)

            # Adaptive inertia weight reduction
            self.inertia_weight = max(self.inertia_weight_min, 
                                      self.inertia_weight - self.alpha * (best_value - min(personal_best_values)) / (best_value + self.epsilon))

            # Contextual self-adaptation strategy
            if np.std(personal_best_values) < self.diversity_threshold:
                perturbation = (ub - lb) * self.alpha * np.random.randn(self.num_particles, self.dim)
                particles = np.clip(particles + perturbation, lb, ub)
                perturbed_values = np.array([func(x) for x in particles])
                eval_count += len(particles)
                
                for j in range(len(particles)):
                    if perturbed_values[j] < best_value:
                        global_best = particles[j]
                        best_value = perturbed_values[j]
                        break

        return global_best