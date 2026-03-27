import numpy as np

class EnhancedDynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize parameters
        num_particles = 12
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        logistic_map_a = 3.8
        logistic_map_x = 0.5
        max_velocity = 0.5
        exploration_phase_threshold = 0.5  # Threshold for switching between phases
        adaptive_mutation_freq = 10  # More frequent perturbations for exploration

        while self.evaluations < self.budget:
            for idx in range(num_particles):
                # Update logistic map for dynamic adjustments
                logistic_map_x = logistic_map_a * logistic_map_x * (1 - logistic_map_x)
                
                # Dynamic inertia weight using logistic map
                inertia = 0.9 - 0.5 * logistic_map_x
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)

                # Cognitive and social components
                cognitive = 2.0 * r1 * (personal_best_positions[idx] - particles[idx])
                neighbor_best_position = self._get_neighbor_best(particles, idx, personal_best_values)
                social = 1.5 * r2 * (neighbor_best_position - particles[idx])
                
                # Update velocity and position
                velocities[idx] = inertia * velocities[idx] + cognitive + social
                velocities[idx] = np.clip(velocities[idx], -max_velocity, max_velocity)
                particles[idx] += velocities[idx]
                particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                # Enhanced adaptive exploration using logistic mapping
                if self.evaluations / self.budget < exploration_phase_threshold:
                    angle = logistic_map_x * np.pi
                    displacement = np.array([
                        max_velocity * np.cos(angle),
                        max_velocity * np.sin(angle)
                    ] + [0] * (self.dim - 2))
                    particles[idx] += displacement[:self.dim]
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                # Evaluate particles
                value = func(particles[idx])
                self.evaluations += 1
                
                # Update personal bests
                if value < personal_best_values[idx]:
                    personal_best_values[idx] = value
                    personal_best_positions[idx] = particles[idx]
                # Update global best
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particles[idx]

                # Adaptive mutation-based perturbation
                if self.evaluations % adaptive_mutation_freq == 0:
                    particles[idx] += np.random.normal(0, 0.05 * (1 - logistic_map_x), self.dim)
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                if self.evaluations >= self.budget:
                    break
            
        return global_best_position, global_best_value

    def _get_neighbor_best(self, particles, idx, personal_best_values):
        """Find the best personal best position among the neighbors."""
        num_particles = len(particles)
        neighborhood_size = 3
        neighbors = [(idx + i) % num_particles for i in range(-neighborhood_size//2, neighborhood_size//2 + 1)]
        best_neighbor_idx = min(neighbors, key=lambda x: personal_best_values[x])
        return particles[best_neighbor_idx]