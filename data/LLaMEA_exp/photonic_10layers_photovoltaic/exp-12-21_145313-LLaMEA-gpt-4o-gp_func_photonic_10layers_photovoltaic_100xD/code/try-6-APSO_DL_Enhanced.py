import numpy as np

class APSO_DL_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, budget // 10)
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.position = np.random.rand(self.num_particles, dim)
        self.velocity = np.random.rand(self.num_particles, dim) * 0.1
        self.best_personal_position = np.copy(self.position)
        self.best_personal_value = np.full(self.num_particles, np.inf)
        self.best_global_position = None
        self.best_global_value = np.inf
        self.evaluations = 0
        self.max_velocity = 0.2 * (func.bounds.ub - func.bounds.lb)

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.num_particles):
                if self.evaluations < self.budget:
                    # Evaluate current position
                    value = func(self.position[i])
                    self.evaluations += 1

                    # Update personal and global bests
                    if value < self.best_personal_value[i]:
                        self.best_personal_value[i] = value
                        self.best_personal_position[i] = self.position[i]
                    if value < self.best_global_value:
                        self.best_global_value = value
                        self.best_global_position = self.position[i]

            # Dynamic adjustment of inertia weight and learning rates
            self.inertia_weight = 0.5 + 0.4 * np.random.rand()
            self.cognitive_coeff = 1.5 + np.random.rand() * 1.5
            self.social_coeff = 1.5 + np.random.rand() * 1.5

            # Update velocity and position
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            self.velocity = (self.inertia_weight * self.velocity +
                             self.cognitive_coeff * r1 * (self.best_personal_position - self.position) +
                             self.social_coeff * r2 * (self.best_global_position - self.position))
            
            # Self-adaptive velocity clamping
            velocity_norm = np.linalg.norm(self.velocity, axis=1)
            clamped_velocity = np.clip(velocity_norm, None, self.max_velocity)
            self.velocity *= clamped_velocity[:, np.newaxis] / velocity_norm[:, np.newaxis]

            self.position += self.velocity

            # Ensure particles are within bounds
            self.position = np.clip(self.position, func.bounds.lb, func.bounds.ub)
            
            # Adaptive restart strategy
            if self.evaluations % (self.budget // 10) == 0:  # Restart 10% of the time
                worst_indices = np.argsort(self.best_personal_value)[-3:]  # Identify a few worst particles
                self.position[worst_indices] = np.random.rand(3, self.dim) * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
                self.velocity[worst_indices] = np.random.rand(3, self.dim) * 0.1
                self.best_personal_value[worst_indices] = np.inf

        return self.best_global_position, self.best_global_value