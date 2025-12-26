import numpy as np
from scipy.stats import levy

class AdaptiveSubswarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_particles = min(40, budget // dim)
        self.num_particles = self.initial_particles
        self.inertia_weight = 0.9
        self.velocity_clamp = 0.15
        self.common_c1 = 1.5
        self.common_c2 = 1.5
        self.best_global_position = np.zeros(dim)
        self.best_global_value = float('inf')
        self.positions = np.random.uniform(0, 1, (self.num_particles, dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.num_particles, float('inf'))
        self.diversity_threshold = 0.1
        self.base_mutation_rate = 0.05
        self.mutation_rate = self.base_mutation_rate
        self.velocity_scaling_factor = 0.05
        self.quantum_scale = 0.1

    def update_velocity(self, particle_idx):
        inertia = self.inertia_weight * self.velocities[particle_idx]
        cognitive = self.adaptive_c1(particle_idx) * np.random.rand(self.dim) * (self.personal_best_positions[particle_idx] - self.positions[particle_idx])
        social = self.adaptive_c2(particle_idx) * np.random.rand(self.dim) * (self.best_global_position - self.positions[particle_idx])
        synergy = np.mean(self.velocities, axis=0)
        exploration_component = levy.rvs(size=self.dim) * self.velocity_scaling_factor
        quantum_boost = np.random.normal(scale=self.quantum_scale, size=self.dim)
        new_velocity = inertia + cognitive + social + 0.1 * synergy + exploration_component + quantum_boost
        np.clip(new_velocity, -self.velocity_clamp, self.velocity_clamp, out=new_velocity)
        return new_velocity

    def update_position(self, particle_idx, func):
        self.velocities[particle_idx] = self.update_velocity(particle_idx)
        self.positions[particle_idx] += self.velocities[particle_idx]
        np.clip(self.positions[particle_idx], func.bounds.lb, func.bounds.ub, out=self.positions[particle_idx])

        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.normal(0, self.velocity_scaling_factor, self.dim)
            self.positions[particle_idx] += mutation_vector

    def calculate_diversity(self):
        mean_position = np.mean(self.positions, axis=0)
        diversity = np.mean(np.linalg.norm(self.positions - mean_position, axis=1))
        return diversity

    def adapt_learning_parameters(self):
        diversity = self.calculate_diversity()
        self.mutation_rate = max(self.base_mutation_rate, self.base_mutation_rate * (diversity / self.diversity_threshold))
        self.velocity_scaling_factor = 0.1 + 0.05 * (diversity / self.diversity_threshold)
        self.quantum_scale = max(0.05, 0.1 * (diversity / self.diversity_threshold))
        if diversity < self.diversity_threshold:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.98)
        else:
            self.inertia_weight = min(0.9, self.inertia_weight * 1.02)

    def adaptive_c1(self, particle_idx):
        local_improvement = (self.personal_best_values[particle_idx] - self.best_global_value) / (self.best_global_value + 1e-6)
        return self.common_c1 * (1 + local_improvement)

    def adaptive_c2(self, particle_idx):
        global_influence = (self.best_global_value - self.personal_best_values[particle_idx]) / (self.personal_best_values[particle_idx] + 1e-6)
        return self.common_c2 * (1 + global_influence)

    def cooperative_local_search(self, func):
        grouped_indices = np.arange(self.num_particles)
        np.random.shuffle(grouped_indices)
        for i in range(0, self.num_particles, 2):
            if i + 1 < self.num_particles:
                idx1, idx2 = grouped_indices[i], grouped_indices[i+1]
                if self.personal_best_values[idx1] < self.personal_best_values[idx2]:
                    offset = levy.rvs(size=self.dim) * self.velocity_scaling_factor
                    self.positions[idx2] = self.personal_best_positions[idx1] + offset
                    np.clip(self.positions[idx2], func.bounds.lb, func.bounds.ub, out=self.positions[idx2])
                else:
                    offset = levy.rvs(size=self.dim) * self.velocity_scaling_factor
                    self.positions[idx1] = self.personal_best_positions[idx2] + offset
                    np.clip(self.positions[idx1], func.bounds.lb, func.bounds.ub, out=self.positions[idx1])

    def dynamic_population_resize(self):
        diversity = self.calculate_diversity()
        if diversity < self.diversity_threshold / 2:
            if self.num_particles > self.initial_particles // 2:
                self.num_particles -= 1
                self.positions = self.positions[:self.num_particles]
                self.velocities = self.velocities[:self.num_particles]
                self.personal_best_positions = self.personal_best_positions[:self.num_particles]
                self.personal_best_values = self.personal_best_values[:self.num_particles]
        else:
            if self.num_particles < self.initial_particles * 1.5:
                self.num_particles += 1
                new_position = np.random.uniform(0, 1, (1, self.dim))
                new_velocity = np.random.uniform(-0.1, 0.1, (1, self.dim))
                self.positions = np.vstack((self.positions, new_position))
                self.velocities = np.vstack((self.velocities, new_velocity))
                self.personal_best_positions = np.vstack((self.personal_best_positions, new_position))
                self.personal_best_values = np.append(self.personal_best_values, float('inf'))

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.num_particles):
                current_value = func(self.positions[i])
                eval_count += 1

                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = 0.7 * self.personal_best_positions[i] + 0.3 * self.positions[i]

                if current_value < self.best_global_value:
                    self.best_global_value = current_value
                    self.best_global_position = self.positions[i]

                if eval_count >= self.budget:
                    break

                self.update_position(i, func)

            self.adapt_learning_parameters()
            self.cooperative_local_search(func)
            self.dynamic_population_resize()

        return self.best_global_position, self.best_global_value