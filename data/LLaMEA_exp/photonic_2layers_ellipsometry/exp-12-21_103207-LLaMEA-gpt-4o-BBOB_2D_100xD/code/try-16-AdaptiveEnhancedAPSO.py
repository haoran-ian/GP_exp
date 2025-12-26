import numpy as np

class AdaptiveEnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, budget // dim)
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocity_clamp = 0.15
        self.best_global_position = np.zeros(dim)
        self.best_global_value = float('inf')
        self.positions = np.random.rand(self.num_particles, dim)
        self.velocities = np.random.rand(self.num_particles, dim) * 0.1
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.num_particles, float('inf'))
        self.diversity_threshold = 0.1
        self.scaling_factor = 0.1

    def update_velocity(self, particle_idx):
        inertia = self.inertia_weight * self.velocities[particle_idx]
        cognitive = self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[particle_idx] - self.positions[particle_idx])
        social = self.c2 * np.random.rand(self.dim) * (self.best_global_position - self.positions[particle_idx])
        exploration_component = np.random.normal(0, self.scaling_factor, self.dim)
        new_velocity = inertia + cognitive + social + exploration_component
        np.clip(new_velocity, -self.velocity_clamp, self.velocity_clamp, out=new_velocity)
        return new_velocity
    
    def update_position(self, particle_idx, func):
        self.velocities[particle_idx] = self.update_velocity(particle_idx)
        self.positions[particle_idx] += self.velocities[particle_idx]
        np.clip(self.positions[particle_idx], func.bounds.lb, func.bounds.ub, out=self.positions[particle_idx])

    def calculate_diversity(self):
        mean_position = np.mean(self.positions, axis=0)
        diversity = np.mean(np.linalg.norm(self.positions - mean_position, axis=1))
        return diversity

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            dynamic_scaling = 1 + self.scaling_factor * np.sin(2 * np.pi * eval_count / self.budget)
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
            
            # Adaptation Mechanism
            success_rate = np.mean(self.personal_best_values < self.best_global_value)
            diversity = self.calculate_diversity()
            
            if success_rate > 0.6:
                self.inertia_weight *= 0.9
                self.c1 *= 1.1
                self.c2 *= 0.9
            elif success_rate < 0.4:
                self.inertia_weight *= 1.1
                self.c1 *= 0.9
                self.c2 *= 1.1

            if diversity < self.diversity_threshold:
                self.inertia_weight *= 1.05
                self.c1 *= 0.95
                self.c2 *= 1.05

            # Apply dynamic scaling to velocity clamp
            self.velocity_clamp *= dynamic_scaling

        return self.best_global_position, self.best_global_value