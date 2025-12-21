import numpy as np

class EnhancedPSOWithHierarchicalDisturbance:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.temperature = 100
        self.cooling_rate = 0.95
        self.diversity_threshold = 0.05
        self.compression_factor = 0.5
        self.disturbance_factor = 0.1
        self.group_size = 10

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(x) for x in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        best_value = min(personal_best_values)

        eval_count = self.num_particles

        while eval_count < self.budget:
            for i in range(0, self.num_particles, self.group_size):
                group_particles = particles[i:i+self.group_size]
                group_best_idx = np.argmin([func(p) for p in group_particles])
                group_best = group_particles[group_best_idx]

                for j in range(min(self.group_size, self.num_particles - i)):
                    r1, r2 = np.random.rand(2)
                    velocities[i+j] = (self.inertia_weight * velocities[i+j] +
                                       self.cognitive_coeff * r1 * (personal_best[i+j] - particles[i+j]) +
                                       self.social_coeff * r2 * (group_best - particles[i+j]))
                    particles[i+j] = np.clip(particles[i+j] + velocities[i+j], lb, ub)

                    current_value = func(particles[i+j])
                    eval_count += 1

                    if current_value < personal_best_values[i+j]:
                        personal_best[i+j] = particles[i+j]
                        personal_best_values[i+j] = current_value

                        if current_value < best_value:
                            global_best = particles[i+j]
                            best_value = current_value

                    if eval_count >= self.budget:
                        break

                # Apply a random disturbance for diversity
                if np.std(personal_best_values[i:i+self.group_size]) < self.diversity_threshold:
                    disturbance = self.disturbance_factor * np.random.randn(*group_particles.shape)
                    group_particles += disturbance
                    group_particles = np.clip(group_particles, lb, ub)
                    for k in range(len(group_particles)):
                        value = func(group_particles[k])
                        eval_count += 1
                        if value < personal_best_values[i+k]:
                            personal_best[i+k] = group_particles[k]
                            personal_best_values[i+k] = value

            # Adaptive inertia weight reduction
            self.inertia_weight = max(self.inertia_weight_min, 
                                      self.inertia_weight * self.cooling_rate + 0.1 * (best_value - min(personal_best_values)))

            self.temperature *= self.cooling_rate

        return global_best