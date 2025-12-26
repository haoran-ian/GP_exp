import numpy as np

class DAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, budget // dim)
        self.inertia_weight = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.velocity_clamp = 0.1
        self.best_global_position = np.zeros(dim)
        self.best_global_value = float('inf')
        self.positions = np.random.rand(self.num_particles, dim)
        self.velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.num_particles, float('inf'))
        self.diversity_threshold = 0.1  # Initial threshold for diversity

    def update_velocity(self, particle_idx, iter_fraction):
        inertia = (0.5 + iter_fraction) * self.velocities[particle_idx]
        cognitive = self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[particle_idx] - self.positions[particle_idx])
        social = self.c2 * np.random.rand(self.dim) * (self.best_global_position - self.positions[particle_idx])
        new_velocity = inertia + cognitive + social
        np.clip(new_velocity, -self.velocity_clamp, self.velocity_clamp, out=new_velocity)
        return new_velocity
    
    def update_position(self, particle_idx, func):
        self.velocities[particle_idx] = self.update_velocity(particle_idx, self.iter_fraction)
        self.positions[particle_idx] += self.velocities[particle_idx]
        np.clip(self.positions[particle_idx], func.bounds.lb, func.bounds.ub, out=self.positions[particle_idx])

    def calculate_diversity(self):
        mean_position = np.mean(self.positions, axis=0)
        diversity = np.mean(np.linalg.norm(self.positions - mean_position, axis=1))
        return diversity

    def opposition_learning(self):
        for i in range(self.num_particles):
            opposite_position = func.bounds.lb + func.bounds.ub - self.positions[i]
            opposite_position = np.clip(opposite_position, func.bounds.lb, func.bounds.ub)
            opposite_value = func(opposite_position)
            if opposite_value < self.personal_best_values[i]:
                self.personal_best_values[i] = opposite_value
                self.personal_best_positions[i] = opposite_position

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            self.iter_fraction = eval_count / self.budget
            self.opposition_learning()
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
            
            # Adaptation Mechanism based on diversity
            diversity = self.calculate_diversity()
            if diversity < self.diversity_threshold:
                self.inertia_weight = min(self.inertia_weight * 1.05, 0.9)
                self.c1 = max(self.c1 * 0.95, 1.5)
                self.c2 = min(self.c2 * 1.05, 2.0)

        return self.best_global_position, self.best_global_value