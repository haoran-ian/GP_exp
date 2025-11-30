import numpy as np

class AGB_PSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.gradient_step_size = 0.1
        self.mutation_factor = 0.8  # DE mutation factor
        self.num_mutations = 3       # Number of differential mutations

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.num_particles, np.inf)
        global_best_position = None
        global_best_value = np.inf
        
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                current_value = func(positions[i])
                evaluations += 1
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = positions[i]
                    
                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = positions[i]
                    
                if evaluations >= self.budget:
                    break
            
            for i in range(self.num_particles):
                cognitive_component = self.cognitive_constant * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.social_constant * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = (self.inertia_weight * velocities[i] + cognitive_component + social_component)
                
                grad_estimate = self.estimate_gradient(func, positions[i])
                velocities[i] += self.gradient_step_size * grad_estimate

                # Differential Evolution Mutation
                idxs = np.random.choice(self.num_particles, 5, replace=False)
                x1, x2, x3 = positions[idxs[:3]]
                mutation_vector = x1 + self.mutation_factor * (x2 - x3)
                positions[i] = positions[i] + velocities[i] + mutation_vector
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
        
        return global_best_position, global_best_value

    def estimate_gradient(self, func, position, epsilon=1e-5):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            perturbed_position = np.copy(position)
            perturbed_position[i] += epsilon
            f_x_plus_eps = func(perturbed_position)
            
            perturbed_position[i] -= 2 * epsilon
            f_x_minus_eps = func(perturbed_position)
            
            grad[i] = (f_x_plus_eps - f_x_minus_eps) / (2 * epsilon)
        
        return grad