import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.num_diffs = 15
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.differential_weight = 0.8
        self.crossover_prob = 0.95  # Line changed: Adjusted crossover probability

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best_idx = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_value = personal_best_values[global_best_idx]

        evaluations = self.num_particles

        while evaluations < self.budget:
            # PSO Update
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                self.inertia_weight = 0.9 - (evaluations / self.budget) * 0.5  # Dynamic inertia weight adjustment
                velocities[i] = (self.inertia_weight * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)
                
                current_value = func(particles[i])
                evaluations += 1
                
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]
                    
                    if current_value < global_best_value:
                        global_best_value = current_value
                        global_best_position = particles[i]

                if evaluations >= self.budget:
                    break

            # DE Update
            if evaluations < self.budget:
                for i in range(self.num_diffs):
                    idxs = np.random.choice(self.num_particles, 3, replace=False)
                    x0, x1, x2 = personal_best_positions[idxs]
                    mutant_vector = np.clip(x0 + self.differential_weight * (x1 - x2), self.lower_bound, self.upper_bound)
                    trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant_vector, personal_best_positions[i])
                    
                    trial_value = func(trial_vector)
                    evaluations += 1
                    
                    if trial_value < personal_best_values[i]:
                        personal_best_values[i] = trial_value
                        personal_best_positions[i] = trial_vector
                        
                        if trial_value < global_best_value:
                            global_best_value = trial_value
                            global_best_position = trial_vector

                    if evaluations >= self.budget:
                        break

        return global_best_position, global_best_value