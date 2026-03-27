import numpy as np

class DynamicSwarmIntelligenceEnhancer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.05
        self.social_coeff = 1.3
        self.inertia_damping = 0.99
        self.local_search_prob = 0.1
        self.global_best_position = None
        self.global_best_value = np.inf

    def __call__(self, func):
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        evals = self.population_size

        min_index = np.argmin(personal_best_values)
        self.global_best_position = personal_best_positions[min_index]
        self.global_best_value = personal_best_values[min_index]
        
        while evals < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i] +
                    self.cognitive_coeff * r1 * (personal_best_positions[i] - particles[i]) +
                    self.social_coeff * r2 * (self.global_best_position - particles[i])
                )
                
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)
                
                if np.random.rand() < self.local_search_prob:
                    local_step = np.random.uniform(-0.1, 0.1, self.dim)
                    particles[i] += local_step
                    particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)
                
                new_value = func(particles[i])
                evals += 1
                
                if new_value < personal_best_values[i]:
                    personal_best_positions[i] = np.copy(particles[i])
                    personal_best_values[i] = new_value
                
                if new_value < self.global_best_value:
                    self.global_best_position = np.copy(particles[i])
                    self.global_best_value = new_value
                
                if evals >= self.budget:
                    break
            
            self.inertia_weight *= self.inertia_damping
        
        return self.global_best_position, self.global_best_value