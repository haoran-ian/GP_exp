import numpy as np

class HybridParticleGeneticOptimizer:
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
        self.mutation_probability = 0.1
        
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
            
            self.inertia_weight = max(self.inertia_weight_min, 
                                      self.inertia_weight * self.cooling_rate + 0.1 * (best_value - min(personal_best_values)))

            if np.random.rand() < np.exp(-abs(current_value - best_value) / self.temperature):
                global_best = particles[i]
                best_value = current_value
            
            self.temperature *= self.cooling_rate
            
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
            
            # Genetic algorithm-inspired crossover and mutation
            if self.diversity_threshold < np.std(personal_best_values) < 0.2 and eval_count + self.num_particles < self.budget:
                idx1, idx2 = np.random.choice(self.num_particles, 2, replace=False)
                crossover_point = np.random.randint(1, self.dim)
                child = np.concatenate((personal_best[idx1][:crossover_point],
                                        personal_best[idx2][crossover_point:]))
                
                if np.random.rand() < self.mutation_probability:
                    mutation_idx = np.random.randint(self.dim)
                    child[mutation_idx] = np.random.uniform(lb[mutation_idx], ub[mutation_idx])
                
                child_value = func(child)
                eval_count += 1

                if child_value < best_value:
                    global_best = child
                    best_value = child_value

        return global_best