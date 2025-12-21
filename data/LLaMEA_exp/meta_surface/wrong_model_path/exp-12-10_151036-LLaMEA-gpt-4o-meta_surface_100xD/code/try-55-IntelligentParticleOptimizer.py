import numpy as np

class IntelligentParticleOptimizer:
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
            
            # Adaptive inertia weight reduction
            self.inertia_weight = max(self.inertia_weight_min, 
                                      self.inertia_weight * self.cooling_rate + 0.1 * (best_value - min(personal_best_values)))

            # Simulated annealing-inspired acceptance mechanism
            if np.random.rand() < np.exp(-abs(current_value - best_value) / self.temperature):
                global_best = particles[i]
                best_value = current_value
            
            self.temperature *= self.cooling_rate
            
            # Enhanced exploration strategy using differential evolution
            if np.std(personal_best_values) < self.diversity_threshold:
                for j in range(self.num_particles):
                    indices = np.random.choice(self.num_particles, 3, replace=False)
                    mutant_vector = personal_best[indices[0]] + 0.8 * (personal_best[indices[1]] - personal_best[indices[2]])
                    trial_vector = np.where(np.random.rand(self.dim) < 0.5, mutant_vector, particles[j])
                    trial_vector = np.clip(trial_vector, lb, ub)
                    trial_value = func(trial_vector)
                    eval_count += 1
                    
                    if trial_value < personal_best_values[j]:
                        personal_best[j] = trial_vector
                        personal_best_values[j] = trial_value
                        
                        if trial_value < best_value:
                            global_best = trial_vector
                            best_value = trial_value
                
                if eval_count >= self.budget:
                    break

        return global_best