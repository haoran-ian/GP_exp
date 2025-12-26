import numpy as np

class RefinedParticleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff_min = 0.5
        self.cognitive_coeff_max = 2.5
        self.social_coeff_min = 0.5
        self.social_coeff_max = 2.5
        self.temperature = 100
        self.cooling_rate = 0.95
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
                dynamic_cognitive_coeff = self.cognitive_coeff_min + (self.cognitive_coeff_max - self.cognitive_coeff_min) * np.random.rand()
                dynamic_social_coeff = self.social_coeff_min + (self.social_coeff_max - self.social_coeff_min) * np.random.rand()
                
                local_best_neighbor = np.argmin(personal_best_values[max(0, i-self.neighborhood_size):min(self.num_particles, i+self.neighborhood_size)])
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 dynamic_cognitive_coeff * r1 * (personal_best[i] - particles[i]) +
                                 dynamic_social_coeff * r2 * (personal_best[local_best_neighbor] - particles[i]))
                
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
            
            # Adaptive inertia weight reduction
            self.inertia_weight = max(self.inertia_weight_min, 
                                      self.inertia_weight * self.cooling_rate)

            self.temperature *= self.cooling_rate
            
            # Enhanced diversity preservation with random exploration
            if np.std(personal_best_values) < self.diversity_threshold:
                exploration_particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
                exploration_values = np.array([func(x) for x in exploration_particles])
                eval_count += len(exploration_particles)
                
                for j in range(len(exploration_particles)):
                    if exploration_values[j] < best_value:
                        global_best = exploration_particles[j]
                        best_value = exploration_values[j]
                        break

        return global_best