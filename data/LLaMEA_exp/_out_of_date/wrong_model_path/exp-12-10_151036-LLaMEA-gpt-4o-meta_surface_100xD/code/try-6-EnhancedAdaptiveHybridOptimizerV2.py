import numpy as np

class EnhancedAdaptiveHybridOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_particles = 50
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.temperature = 100
        self.cooling_rate = 0.95
        self.diversity_threshold = 0.1
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = self.initial_particles
        particles = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
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
                # Introduce velocity clamping for more stable convergence
                velocities[i] = np.clip(velocities[i], -0.1 * (ub - lb), 0.1 * (ub - lb))
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
            
            # Improved diversity and population size adjustment
            if np.std(personal_best_values) < self.diversity_threshold:
                additional_particles = np.random.uniform(lb, ub, (num_particles // 5, self.dim))
                additional_values = np.array([func(x) for x in additional_particles])
                eval_count += len(additional_particles)
                
                for j, val in enumerate(additional_values):
                    if val < best_value or np.random.rand() < 0.1:
                        global_best = additional_particles[j]
                        best_value = val
                        break
                # Increase population size adaptively to escape stagnation
                num_particles = min(num_particles + num_particles // 10, self.budget - eval_count)
                if eval_count >= self.budget:
                    break

        return global_best