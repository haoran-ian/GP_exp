import numpy as np

class ImprovedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_num_particles = 50
        self.max_num_particles = 100
        self.min_num_particles = 20
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.temperature = 100
        self.cooling_rate = 0.95
        self.diversity_threshold = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = self.init_num_particles
        particles = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.zeros((num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(x) for x in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        best_value = min(personal_best_values)

        eval_count = num_particles
        
        while eval_count < self.budget:
            for i in range(num_particles):
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
            
            # Dynamic inertia weight adaptation
            self.inertia_weight = max(self.inertia_weight_min, self.inertia_weight * self.cooling_rate)
            
            # Simulated annealing-inspired temperature adjustment
            if np.random.rand() < np.exp(-abs(current_value - best_value) / self.temperature):
                global_best = particles[i]
                best_value = current_value
            
            self.temperature *= self.cooling_rate
            
            # Improved diversity preservation and population adjustment mechanism
            if np.std(personal_best_values) < self.diversity_threshold:
                random_particles = np.random.uniform(lb, ub, (num_particles // 5, self.dim))
                random_values = np.array([func(x) for x in random_particles])
                eval_count += len(random_particles)
                
                for j in range(len(random_particles)):
                    if random_values[j] < best_value or np.random.rand() < 0.1:
                        global_best = random_particles[j]
                        best_value = random_values[j]
                        break

                # Adaptive population resizing
                num_particles = min(self.max_num_particles, max(self.min_num_particles, num_particles + 5))
                particles = np.random.uniform(lb, ub, (num_particles, self.dim))
                velocities = np.zeros((num_particles, self.dim))
                personal_best = particles.copy()
                personal_best_values = np.array([func(x) for x in personal_best])
                eval_count += num_particles - len(personal_best_values)
        
        return global_best