import numpy as np
from scipy.stats import levy

class MultiSwarmEnhancedSynergisticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarms = min(5, budget // (dim * 10))
        self.num_swarms = self.initial_swarms
        self.particles_per_swarm = min(40, budget // (dim * self.num_swarms))
        self.inertia_weight = 0.9
        self.c1 = 2.0
        self.c2 = 1.0
        self.velocity_clamp = 0.15
        self.global_best_positions = [np.zeros(dim) for _ in range(self.num_swarms)]
        self.global_best_values = [float('inf') for _ in range(self.num_swarms)]
        self.positions = [np.random.uniform(0, 1, (self.particles_per_swarm, dim)) for _ in range(self.num_swarms)]
        self.velocities = [np.random.uniform(-0.1, 0.1, (self.particles_per_swarm, dim)) for _ in range(self.num_swarms)]
        self.personal_best_positions = [np.copy(pos) for pos in self.positions]
        self.personal_best_values = [np.full(self.particles_per_swarm, float('inf')) for _ in range(self.num_swarms)]
        self.diversity_threshold = 0.1
        self.base_mutation_rate = 0.05
        self.mutation_rate = self.base_mutation_rate
        self.velocity_scaling_factor = 0.05
        self.quantum_scale = 0.1

    def update_velocity(self, swarm_idx, particle_idx):
        inertia = self.inertia_weight * self.velocities[swarm_idx][particle_idx]
        cognitive = self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[swarm_idx][particle_idx] - self.positions[swarm_idx][particle_idx])
        social = self.c2 * np.random.rand(self.dim) * (self.global_best_positions[swarm_idx] - self.positions[swarm_idx][particle_idx])
        synergy = np.mean(self.velocities[swarm_idx], axis=0)
        exploration_component = levy.rvs(size=self.dim) * self.velocity_scaling_factor
        quantum_boost = np.random.normal(scale=self.quantum_scale, size=self.dim)
        new_velocity = inertia + cognitive + social + 0.1 * synergy + exploration_component + quantum_boost
        np.clip(new_velocity, -self.velocity_clamp, self.velocity_clamp, out=new_velocity)
        return new_velocity

    def update_position(self, swarm_idx, particle_idx, func):
        self.velocities[swarm_idx][particle_idx] = self.update_velocity(swarm_idx, particle_idx)
        self.positions[swarm_idx][particle_idx] += self.velocities[swarm_idx][particle_idx]
        np.clip(self.positions[swarm_idx][particle_idx], func.bounds.lb, func.bounds.ub, out=self.positions[swarm_idx][particle_idx])

        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.normal(0, self.velocity_scaling_factor, self.dim)
            self.positions[swarm_idx][particle_idx] += mutation_vector

    def calculate_diversity(self, swarm_idx):
        mean_position = np.mean(self.positions[swarm_idx], axis=0)
        diversity = np.mean(np.linalg.norm(self.positions[swarm_idx] - mean_position, axis=1))
        return diversity

    def adapt_learning_parameters(self, swarm_idx):
        diversity = self.calculate_diversity(swarm_idx)
        self.mutation_rate = max(self.base_mutation_rate, self.base_mutation_rate * (diversity / self.diversity_threshold))
        self.velocity_scaling_factor = 0.1 + 0.05 * (diversity / self.diversity_threshold)
        self.quantum_scale = max(0.05, 0.1 * (diversity / self.diversity_threshold))
        if diversity < self.diversity_threshold:
            self.c1 = min(3.0, self.c1 * 1.1)
            self.c2 = max(0.5, self.c2 * 0.9)
            self.inertia_weight = max(0.4, self.inertia_weight * 0.98)
        else:
            self.c1 = max(0.5, self.c1 * 0.9)
            self.c2 = min(3.0, self.c2 * 1.1)
            self.inertia_weight = min(0.9, self.inertia_weight * 1.02)

    def cooperative_local_search(self, swarm_idx, func):
        for i in range(self.particles_per_swarm):
            for j in range(i + 1, self.particles_per_swarm):
                if self.personal_best_values[swarm_idx][j] < self.personal_best_values[swarm_idx][i]:
                    offset = levy.rvs(size=self.dim) * self.velocity_scaling_factor
                    self.positions[swarm_idx][i] = self.personal_best_positions[swarm_idx][j] + offset
                    np.clip(self.positions[swarm_idx][i], func.bounds.lb, func.bounds.ub, out=self.positions[swarm_idx][i])
                elif self.personal_best_values[swarm_idx][i] < self.personal_best_values[swarm_idx][j]:
                    offset = levy.rvs(size=self.dim) * self.velocity_scaling_factor
                    self.positions[swarm_idx][j] = self.personal_best_positions[swarm_idx][i] + offset
                    np.clip(self.positions[swarm_idx][j], func.bounds.lb, func.bounds.ub, out=self.positions[swarm_idx][j])

    def adaptive_interaction_between_swarms(self, func):
        for swarm_idx in range(self.num_swarms):
            for other_swarm_idx in range(self.num_swarms):
                if swarm_idx != other_swarm_idx:
                    if self.global_best_values[other_swarm_idx] < self.global_best_values[swarm_idx]:
                        self.global_best_positions[swarm_idx] = self.global_best_positions[other_swarm_idx]
                        self.global_best_values[swarm_idx] = self.global_best_values[other_swarm_idx]

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            for swarm_idx in range(self.num_swarms):
                for i in range(self.particles_per_swarm):
                    current_value = func(self.positions[swarm_idx][i])
                    eval_count += 1

                    if current_value < self.personal_best_values[swarm_idx][i]:
                        self.personal_best_values[swarm_idx][i] = current_value
                        self.personal_best_positions[swarm_idx][i] = self.positions[swarm_idx][i]

                    if current_value < self.global_best_values[swarm_idx]:
                        self.global_best_values[swarm_idx] = current_value
                        self.global_best_positions[swarm_idx] = self.positions[swarm_idx][i]

                    if eval_count >= self.budget:
                        break

                    self.update_position(swarm_idx, i, func)

                self.adapt_learning_parameters(swarm_idx)
                self.cooperative_local_search(swarm_idx, func)

            self.adaptive_interaction_between_swarms(func)

        best_overall_value = min(self.global_best_values)
        best_overall_position = self.global_best_positions[self.global_best_values.index(best_overall_value)]
        return best_overall_position, best_overall_value