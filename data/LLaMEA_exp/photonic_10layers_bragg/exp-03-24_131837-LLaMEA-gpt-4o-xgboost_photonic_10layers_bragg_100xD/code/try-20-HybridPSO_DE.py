import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.2
        self.cognitive_coeff_initial = 2.0
        self.social_coeff_initial = 1.2
        self.cognitive_coeff_final = 0.5
        self.social_coeff_final = 2.5
        self.global_best_position = None
        self.global_best_value = np.inf
        self.mutation_factor = 0.8  # DE Mutation factor
        self.crossover_prob = 0.9  # DE Crossover probability

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
            )
            cognitive_coeff = self.cognitive_coeff_initial - (
                (self.cognitive_coeff_initial - self.cognitive_coeff_final) * (evals / self.budget) ** 0.7
            )
            social_coeff = self.social_coeff_initial + (
                (self.social_coeff_final - self.social_coeff_initial) * (evals / self.budget) ** 1.3
            )

            # Hybrid PSO and DE
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    inertia_weight * velocities[i] +
                    cognitive_coeff * r1 * (personal_best_positions[i] - particles[i]) +
                    social_coeff * r2 * (self.global_best_position - particles[i])
                )

                # Differential Evolution Mutation and Crossover
                if np.random.rand() < self.crossover_prob:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = particles[indices[0]], particles[indices[1]], particles[indices[2]]
                    mutant_vector = x0 + self.mutation_factor * (x1 - x2)
                    trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant_vector, particles[i])
                    trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
                    trial_value = func(trial_vector)
                    evals += 1
                    if trial_value < personal_best_values[i]:
                        personal_best_positions[i] = trial_vector
                        personal_best_values[i] = trial_value
                        if trial_value < self.global_best_value:
                            self.global_best_position = trial_vector
                            self.global_best_value = trial_value

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