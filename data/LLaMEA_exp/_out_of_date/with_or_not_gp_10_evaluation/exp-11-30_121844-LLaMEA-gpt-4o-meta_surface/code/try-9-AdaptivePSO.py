import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim, swarm_size=30, inertia_bounds=(0.4, 0.9), c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia_bounds = inertia_bounds
        self.c1 = c1
        self.c2 = c2
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (swarm_size, dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (swarm_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(swarm_size, float('inf'))
        self.global_best_position = None
        self.global_best_value = float('inf')

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            # Evaluate fitness
            for i in range(self.swarm_size):
                fitness = func(self.positions[i])
                eval_count += 1
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.positions[i]
            
            # Adaptive inertia weight
            inertia_weight = self.inertia_bounds[1] - ((self.inertia_bounds[1] - self.inertia_bounds[0]) * (eval_count / self.budget))
            
            # Update velocities and positions with dynamic learning factors
            for i in range(self.swarm_size):
                c1_decay = self.c1 * (1 - eval_count / self.budget)
                c2_growth = self.c2 * (eval_count / self.budget)
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = c1_decay * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = c2_growth * r2 * (self.global_best_position - self.positions[i])
                stochastic_perturbation = np.random.normal(0, 0.1, self.dim)
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity + stochastic_perturbation

                # Clamp velocities to prevent excessive movement
                velocity_bound = 1.0 - (eval_count / self.budget) * 0.5
                self.velocities[i] = np.clip(self.velocities[i], -velocity_bound, velocity_bound)

                # Update positions
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Introduce a decay mechanism in swarm size for faster convergence
            if eval_count > self.budget * 0.5:
                self.swarm_size = max(1, int(self.swarm_size * 0.8))  # Changed rate

        return self.global_best_position, self.global_best_value