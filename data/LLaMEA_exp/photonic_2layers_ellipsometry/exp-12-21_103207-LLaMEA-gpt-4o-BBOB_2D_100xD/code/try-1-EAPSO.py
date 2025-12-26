import numpy as np

class EAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, budget // dim)
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocity_clamp = 0.1
        self.best_global_position = np.zeros(dim)
        self.best_global_value = float('inf')
        self.positions = np.random.rand(self.num_particles, dim)
        self.velocities = np.random.rand(self.num_particles, dim) * 0.1
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.num_particles, float('inf'))
        self.neighborhood_size = max(2, int(0.1 * self.num_particles))
    
    def constriction_coefficient(self, phi):
        return 2 / abs(2 - phi - np.sqrt(phi**2 - 4*phi))
    
    def update_velocity(self, particle_idx, neighbors_best):
        phi = self.c1 + self.c2
        k = self.constriction_coefficient(phi)
        inertia = self.inertia_weight * self.velocities[particle_idx]
        cognitive = self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[particle_idx] - self.positions[particle_idx])
        social = self.c2 * np.random.rand(self.dim) * (neighbors_best - self.positions[particle_idx])
        new_velocity = k * (inertia + cognitive + social)
        np.clip(new_velocity, -self.velocity_clamp, self.velocity_clamp, out=new_velocity)
        return new_velocity
    
    def update_position(self, particle_idx, func):
        neighbors = self.get_neighbors(particle_idx)
        neighbors_best = min(neighbors, key=lambda x: self.personal_best_values[x])
        neighbors_best_position = self.personal_best_positions[neighbors_best]
        self.velocities[particle_idx] = self.update_velocity(particle_idx, neighbors_best_position)
        self.positions[particle_idx] += self.velocities[particle_idx]
        np.clip(self.positions[particle_idx], func.bounds.lb, func.bounds.ub, out=self.positions[particle_idx])
    
    def get_neighbors(self, particle_idx):
        # Use ring topology for neighborhood
        neighbors = [(particle_idx + i) % self.num_particles for i in range(-self.neighborhood_size // 2, self.neighborhood_size // 2 + 1)]
        return neighbors
    
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
            
            # Adaptation Mechanism
            success_rate = np.mean(self.personal_best_values < self.best_global_value)
            if success_rate > 0.6:
                self.inertia_weight *= 0.9
            elif success_rate < 0.4:
                self.inertia_weight *= 1.1
            
        return self.best_global_position, self.best_global_value