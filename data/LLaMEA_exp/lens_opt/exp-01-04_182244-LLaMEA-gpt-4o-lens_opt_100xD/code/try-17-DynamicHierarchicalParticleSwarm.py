import numpy as np

class DynamicHierarchicalParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize parameters
        num_particles = 10
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        
        hierarchy_levels = 3
        inertia_weights = np.linspace(0.9, 0.4, hierarchy_levels)
        hierarchy_indices = np.linspace(0, num_particles - 1, hierarchy_levels, dtype=int)
        
        while self.evaluations < self.budget:
            for idx in range(num_particles):
                # Determine the hierarchy level
                level = np.searchsorted(hierarchy_indices, idx)
                inertia = inertia_weights[level]
                
                # Update velocity and position using hierarchical PSO formula
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = 2.0 * r1 * (personal_best_positions[idx] - particles[idx])
                social = 1.5 * r2 * (global_best_position - particles[idx])
                velocities[idx] = inertia * velocities[idx] + cognitive + social
                particles[idx] += velocities[idx]
                particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                # Evaluate particles
                value = func(particles[idx])
                self.evaluations += 1

                # Update personal and global bests
                if value < personal_best_values[idx]:
                    personal_best_values[idx] = value
                    personal_best_positions[idx] = particles[idx]
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particles[idx]

                # Dynamic interaction based on performance
                if self.evaluations % 50 == 0 and idx == global_best_index:
                    velocities[idx] *= 0.8  # Slow down best particle for fine-tuning
                
                if self.evaluations >= self.budget:
                    break

            # Reassign hierarchy dynamically
            sorted_indices = np.argsort(personal_best_values)
            hierarchy_indices = np.array_split(sorted_indices, hierarchy_levels)

        return global_best_position, global_best_value