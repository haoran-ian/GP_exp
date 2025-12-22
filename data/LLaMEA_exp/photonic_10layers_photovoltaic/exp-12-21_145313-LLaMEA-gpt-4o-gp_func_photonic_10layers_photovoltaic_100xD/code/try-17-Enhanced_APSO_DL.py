import numpy as np

class Enhanced_APSO_DL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, budget // 10)
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.mutation_prob = 0.1
        self.position = np.random.rand(self.num_particles, dim)
        self.velocity = np.random.rand(self.num_particles, dim) * 0.1
        self.best_personal_position = np.copy(self.position)
        self.best_personal_value = np.full(self.num_particles, np.inf)
        self.best_global_position = None
        self.best_global_value = np.inf
        self.evaluations = 0

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
            self.inertia_weight = 0.4 + 0.5 * (1 - self.evaluations / self.budget)
            self.cognitive_coeff = 1.5 + np.random.rand() * 1.5
            self.social_coeff = 1.5 + np.random.rand() * 1.5
            self.mutation_prob = 0.05 + 0.45 * (1 - self.evaluations / self.budget)  # Dynamically adjust mutation probability

            # Update velocity and position using hybrid exploration strategy
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            self.velocity = (self.inertia_weight * self.velocity +
                             self.cognitive_coeff * r1 * (self.best_personal_position - self.position) +
                             self.social_coeff * r2 * (self.best_global_position - self.position))

            # Apply differential mutation to some particles
            for i in range(self.num_particles):
                if np.random.rand() < self.mutation_prob:
                    indices = np.random.choice(self.num_particles, 3, replace=False)
                    F = np.random.rand()
                    mutant = self.position[indices[0]] + F * (self.position[indices[1]] - self.position[indices[2]])
                    self.position[i] = np.clip(mutant, func.bounds.lb, func.bounds.ub)

            self.position += self.velocity

            # Ensure particles are within bounds
            self.position = np.clip(self.position, func.bounds.lb, func.bounds.ub)
            
            # Adaptive restart strategy
            if self.evaluations % (self.budget // 8) == 0:  # Restart 12.5% of the time (changed from 10%)
                worst_indices = np.argsort(self.best_personal_value)[-3:]  # Identify a few worst particles
                self.position[worst_indices] = np.random.rand(3, self.dim) * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
                self.velocity[worst_indices] = np.random.rand(3, self.dim) * 0.1
                self.best_personal_value[worst_indices] = np.inf

        return self.best_global_position, self.best_global_value