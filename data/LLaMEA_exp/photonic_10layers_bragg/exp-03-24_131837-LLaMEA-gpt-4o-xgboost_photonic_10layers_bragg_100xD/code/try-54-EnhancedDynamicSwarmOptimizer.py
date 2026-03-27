import numpy as np

class EnhancedDynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(50, self.budget // 10)
        self.current_population_size = self.initial_population_size
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.2
        self.cognitive_coeff_initial = 2.0
        self.social_coeff_initial = 1.2
        self.cognitive_coeff_final = 0.5
        self.social_coeff_final = 2.5
        self.global_best_position = None
        self.global_best_value = np.inf
        self.mutation_rate_initial = 0.1
        self.mutation_rate_final = 0.05

    def __call__(self, func):
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.initial_population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.initial_population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        evals = self.initial_population_size

        min_index = np.argmin(personal_best_values)
        self.global_best_position = personal_best_positions[min_index]
        self.global_best_value = personal_best_values[min_index]
        
        while evals < self.budget:
            if evals > (self.budget / 2):
                diversity = np.std(personal_best_values)
                self.current_population_size = max(self.initial_population_size // 2, 
                                                   int(self.initial_population_size * (diversity / np.mean(personal_best_values))))
            
            inertia_weight = self.inertia_weight_initial - (
                (self.inertia_weight_initial - self.inertia_weight_final) * (evals / self.budget) ** 2.0)  # Modified decay factor
            cognitive_coeff = self.cognitive_coeff_initial - (
                (self.cognitive_coeff_initial - self.cognitive_coeff_final) * (evals / self.budget) ** 0.7)
            social_coeff = self.social_coeff_initial + (
                (self.social_coeff_final - self.social_coeff_initial) * (evals / self.budget) ** 1.3)
            mutation_rate = self.mutation_rate_initial - (
                (self.mutation_rate_initial - self.mutation_rate_final) * (evals / self.budget))

            if self.global_best_value < np.mean(personal_best_values):
                mutation_rate *= 1.2  # Adjusted mutation rate multiplier

            for i in range(self.current_population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    inertia_weight * velocities[i] +
                    cognitive_coeff * r1 * (personal_best_positions[i] - particles[i]) +
                    social_coeff * r2 * (self.global_best_position - particles[i])
                )
                
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)
                
                if np.random.rand() < mutation_rate:
                    mutation_vector = np.random.uniform(-1, 1, self.dim)
                    particles[i] += mutation_vector * (func.bounds.ub - func.bounds.lb) * mutation_rate
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