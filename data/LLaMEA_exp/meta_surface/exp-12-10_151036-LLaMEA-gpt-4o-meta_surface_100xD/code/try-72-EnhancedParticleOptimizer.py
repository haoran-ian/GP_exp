import numpy as np

class EnhancedParticleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.learning_rate = 0.1
        self.diversity_threshold = 0.1
        self.neighborhood_size = 5
    
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
                particles[i] = np.clip(particles[i] + self.learning_rate * velocities[i], lb, ub)

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
            
            # Adaptive inertia weight reduction
            self.inertia_weight = max(self.inertia_weight_min, 
                                      self.inertia_weight * 0.95 + 0.1 * (best_value - min(personal_best_values)))

            # Adaptive learning rate adjustment
            self.learning_rate = max(0.01, (best_value - min(personal_best_values)) / best_value)
            
            # Elite reinitialization for diversity
            if np.std(personal_best_values) < self.diversity_threshold:
                elite_indices = np.argsort(personal_best_values)[:self.neighborhood_size]
                for idx in elite_indices:
                    particles[idx] = np.random.uniform(lb, ub, self.dim)
                    eval_count += 1

        return global_best