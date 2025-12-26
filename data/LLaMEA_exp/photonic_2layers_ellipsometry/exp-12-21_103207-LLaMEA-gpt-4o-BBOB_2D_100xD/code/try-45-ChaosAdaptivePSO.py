import numpy as np
from scipy.stats import levy

class ChaosAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_particles = min(40, budget // dim)
        self.num_particles = self.initial_particles
        self.inertia_weight = 0.9
        self.c1 = 2.0
        self.c2 = 1.0
        self.velocity_clamp = 0.15
        self.best_global_position = np.zeros(dim)
        self.best_global_value = float('inf')
        self.positions = np.random.uniform(0, 1, (self.num_particles, dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.num_particles, float('inf'))
        self.diversity_threshold = 0.1
        self.scaling_factor = 0.1
        self.lambda_chaos = 3.8  # Logistic map parameter for chaos
        self.chaotic_seq = np.random.rand(self.num_particles)
        self.variance_reduction_factor = 0.99

    def update_velocity(self, particle_idx):
        inertia = self.inertia_weight * self.velocities[particle_idx]
        cognitive = self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[particle_idx] - self.positions[particle_idx])
        social = self.c2 * np.random.rand(self.dim) * (self.best_global_position - self.positions[particle_idx])
        exploration_component = levy.rvs(size=self.dim) * self.scaling_factor
        new_velocity = inertia + cognitive + social + exploration_component
        np.clip(new_velocity, -self.velocity_clamp, self.velocity_clamp, out=new_velocity)
        return new_velocity

    def update_position(self, particle_idx, func):
        self.velocities[particle_idx] = self.update_velocity(particle_idx)
        self.positions[particle_idx] += self.velocities[particle_idx]
        np.clip(self.positions[particle_idx], func.bounds.lb, func.bounds.ub, out=self.positions[particle_idx])

        chaotic_mutation = self.chaotic_seq[particle_idx] * self.scaling_factor
        self.positions[particle_idx] += chaotic_mutation

    def calculate_diversity(self):
        mean_position = np.mean(self.positions, axis=0)
        diversity = np.mean(np.linalg.norm(self.positions - mean_position, axis=1))
        return diversity

    def adapt_learning_parameters(self):
        diversity = self.calculate_diversity()
        self.scaling_factor *= self.variance_reduction_factor
        self.chaotic_seq = self.lambda_chaos * self.chaotic_seq * (1 - self.chaotic_seq)  # Update chaotic sequence
        self.inertia_weight = max(0.4, self.inertia_weight * 0.98 if diversity < self.diversity_threshold else self.inertia_weight * 1.02)
        self.c1, self.c2 = (min(3.0, self.c1 * 1.1), max(0.5, self.c2 * 0.9)) if diversity < self.diversity_threshold else (max(0.5, self.c1 * 0.9), min(3.0, self.c2 * 1.1))

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.num_particles):
                current_value = func(self.positions[i])
                eval_count += 1

                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.positions[i]

                if current_value < self.best_global_value:
                    self.best_global_value = current_value
                    self.best_global_position = self.positions[i]

                if eval_count >= self.budget:
                    break

                self.update_position(i, func)

            self.adapt_learning_parameters()

        return self.best_global_position, self.best_global_value