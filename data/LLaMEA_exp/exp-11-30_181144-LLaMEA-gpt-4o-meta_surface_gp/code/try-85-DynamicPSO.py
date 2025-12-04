import numpy as np

class DynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.c1 = 2.0  # Cognitive parameter
        self.c2 = 2.0  # Social parameter
        self.w_max = 0.9
        self.w_min = 0.4
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.func_evals = 0

    def __call__(self, func):
        eval_budget = self.budget // self.num_particles
        for t in range(eval_budget):
            w = self.w_max - (self.w_max - self.w_min) * (t / eval_budget)
            for i in range(self.num_particles):
                if self.func_evals >= self.budget:
                    break
                value = func(self.positions[i])
                self.func_evals += 1
                if value < self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.positions[i]
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.positions[i]
            
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component
                self.velocities[i] *= 0.9  # Adaptive velocity scaling
                self.velocities[i] = np.clip(self.velocities[i], -2, 2)  # Velocity clamping
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)
            
            # Local search around global best
            if t % 5 == 0:  # Increased local search frequency
                perturbation = np.random.normal(0, 0.1, self.dim)
                candidate_position = np.clip(self.global_best_position + perturbation, self.lower_bound, self.upper_bound)
                candidate_value = func(candidate_position)
                self.func_evals += 1
                if candidate_value < self.global_best_value:
                    self.global_best_value = candidate_value
                    self.global_best_position = candidate_position

        return self.global_best_position, self.global_best_value