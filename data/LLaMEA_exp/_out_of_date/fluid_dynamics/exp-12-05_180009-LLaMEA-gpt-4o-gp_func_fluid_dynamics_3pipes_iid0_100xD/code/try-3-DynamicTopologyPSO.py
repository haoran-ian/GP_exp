import numpy as np

class DynamicTopologyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.num_particles = 20
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, dim))
        self.positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.num_particles, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.function_evaluations = 0
        self.topology_change_interval = 50
        self.neighborhood_size = 4  # Increased neighborhood size

    def evaluate(self, func, particle_index):
        current_value = func(self.positions[particle_index])
        self.function_evaluations += 1
        
        # Update personal best
        if current_value < self.personal_best_values[particle_index]:
            self.personal_best_values[particle_index] = current_value
            self.personal_best_positions[particle_index] = self.positions[particle_index]
        
        # Update global best
        if current_value < self.global_best_value:
            self.global_best_value = current_value
            self.global_best_position = self.positions[particle_index]

    def update_topology(self):
        # Randomly assign neighbors based on the interval
        if self.function_evaluations % self.topology_change_interval == 0:
            self.neighbors = [
                np.random.choice(
                    np.delete(np.arange(self.num_particles), i), 
                    self.neighborhood_size, 
                    replace=False
                )
                for i in range(self.num_particles)
            ]

    def update_velocities_and_positions(self):
        for i in range(self.num_particles):
            neighborhood_best_position = self.personal_best_positions[self.neighbors[i]].min(axis=0)
            inertia_weight = 0.7 + np.random.rand() / 5  # Changed line: Enhanced inertia weight range
            cognitive_component = 2 * np.random.rand() * (self.personal_best_positions[i] - self.positions[i])
            social_component = 2 * np.random.rand() * (neighborhood_best_position - self.positions[i])
            self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_component + social_component
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

    def __call__(self, func):
        while self.function_evaluations < self.budget:
            self.update_topology()
            for i in range(self.num_particles):
                if self.function_evaluations < self.budget:
                    self.evaluate(func, i)
            self.update_velocities_and_positions()
        return self.global_best_position, self.global_best_value