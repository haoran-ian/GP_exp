import numpy as np

class HybridParticleOptimizer:
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
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.learning_rate_adapt = 0.05
        
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
                
                # Periodic velocity perturbation
                if eval_count % (self.num_particles * 2) == 0:
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    velocities[i] += perturbation

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
                                      self.inertia_weight * self.cooling_rate + self.learning_rate_adapt * (best_value - min(personal_best_values)))

            # Simulated annealing-inspired acceptance mechanism
            if np.random.rand() < np.exp(-abs(current_value - best_value) / self.temperature):
                global_best = particles[i]
                best_value = current_value
            
            self.temperature *= self.cooling_rate
            
            # Genetic algorithm-inspired improvements
            if np.std(personal_best_values) < self.diversity_threshold:
                for j in range(self.num_particles):
                    if np.random.rand() < self.mutation_rate:
                        mutation = np.random.normal(0, 0.1, self.dim)
                        particles[j] += mutation

                    if np.random.rand() < self.crossover_rate:
                        partner_idx = np.random.randint(self.num_particles)
                        crossover_point = np.random.randint(1, self.dim)
                        particles[j][:crossover_point] = personal_best[partner_idx][:crossover_point]
                        
                        particles[j] = np.clip(particles[j], lb, ub)

                particles_values = np.array([func(x) for x in particles])
                eval_count += len(particles)
                
                best_idx = np.argmin(particles_values)
                if particles_values[best_idx] < best_value:
                    global_best = particles[best_idx]
                    best_value = particles_values[best_idx]

        return global_best