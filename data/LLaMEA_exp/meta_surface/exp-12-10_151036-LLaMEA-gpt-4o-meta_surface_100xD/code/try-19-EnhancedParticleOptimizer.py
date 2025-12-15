import numpy as np

class EnhancedParticleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.inertia_weight_max = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.diversity_threshold = 0.1
        self.multi_leader_count = 5
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(x) for x in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        multi_leaders = personal_best[np.argsort(personal_best_values)[:self.multi_leader_count]]
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
            
            # Dynamic inertia weight adjustment based on diversity
            self.inertia_weight = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (eval_count / self.budget)
            
            # Multi-leader selection
            if np.std(personal_best_values) < self.diversity_threshold:
                indices = np.random.choice(self.num_particles, self.multi_leader_count)
                multi_leaders = personal_best[indices]
                global_best = multi_leaders[np.argmin([func(x) for x in multi_leaders])]
                best_value = func(global_best)

        return global_best