import numpy as np

class RefinedParticleOptimizer:
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
        self.memory_coeff = 0.2
        self.neighborhood_size = 5
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(x) for x in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        best_value = min(personal_best_values)
        memory = np.zeros((self.num_particles, self.dim))

        eval_count = self.num_particles
        
        while eval_count < self.budget:
            for i in range(self.num_particles):
                # Select neighborhood
                neighbors_idx = np.random.choice(self.num_particles, self.neighborhood_size, replace=False)
                local_best = personal_best[neighbors_idx[np.argmin(personal_best_values[neighbors_idx])]]

                r1, r2, r3 = np.random.rand(3)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best[i] - particles[i]) +
                                 self.social_coeff * r2 * (local_best - particles[i]) +
                                 self.memory_coeff * r3 * memory[i])
                
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                current_value = func(particles[i])
                eval_count += 1
                
                if current_value < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = current_value
                    memory[i] = particles[i] - personal_best[i]
                    
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
            
            # Enhanced diversity preservation with memory influence
            if np.std(personal_best_values) < self.diversity_threshold:
                memory_influenced_particles = particles + memory + np.random.randn(*particles.shape)
                memory_influenced_particles = np.clip(memory_influenced_particles, lb, ub)
                memory_influenced_values = np.array([func(x) for x in memory_influenced_particles])
                eval_count += len(memory_influenced_particles)
                
                for j in range(len(memory_influenced_particles)):
                    if memory_influenced_values[j] < best_value or np.random.rand() < 0.1:
                        global_best = memory_influenced_particles[j]
                        best_value = memory_influenced_values[j]
                        break

        return global_best