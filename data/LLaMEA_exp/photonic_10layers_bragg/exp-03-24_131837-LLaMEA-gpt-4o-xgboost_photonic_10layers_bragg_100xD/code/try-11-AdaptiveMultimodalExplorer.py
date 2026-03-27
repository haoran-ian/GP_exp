import numpy as np

class AdaptiveMultimodalExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.2  # Adjusted for deeper exploration
        self.cognitive_coeff = 2.0
        self.social_coeff = 1.2
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
            inertia_weight = self.inertia_weight_initial - (
                (self.inertia_weight_initial - self.inertia_weight_final) * (evals / self.budget) ** 1.5
            )  # Modified inertia adjustment for more adaptive search

            for i in range(self.population_size):
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