import numpy as np

class AdaptiveMultimodalExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.3  # Adjusted for improved exploitation
        self.cognitive_coeff = 2.0
        self.social_coeff = 1.5  # Slightly increased to enhance social learning
        self.velocity_scale = 0.1  # New parameter for scaling velocity
        self.local_search_prob = 0.1  # Probability of performing local search
        self.global_best_position = None
        self.global_best_value = np.inf

    def __call__(self, func):
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * self.velocity_scale
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        evals = self.population_size

        min_index = np.argmin(personal_best_values)
        self.global_best_position = personal_best_positions[min_index]
        self.global_best_value = personal_best_values[min_index]
        
        while evals < self.budget:
            inertia_weight = self.inertia_weight_initial - (
                (self.inertia_weight_initial - self.inertia_weight_final) * evals / self.budget
            )  # Dynamic inertia weight

            for i in range(self.population_size):
                if np.random.rand() < self.local_search_prob:
                    particles[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)  # Local search step
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    inertia_weight * velocities[i] +
                    self.cognitive_coeff * r1 * (personal_best_positions[i] - particles[i]) +
                    self.social_coeff * r2 * (self.global_best_position - particles[i])
                )
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)
                
                new_value = func(particles[i])
                evals += 1
                
                if new_value < personal_best_values[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_values[i] = new_value
                
                if new_value < self.global_best_value:
                    self.global_best_position = particles[i]
                    self.global_best_value = new_value

                if evals >= self.budget:
                    break
        
        return self.global_best_position, self.global_best_value