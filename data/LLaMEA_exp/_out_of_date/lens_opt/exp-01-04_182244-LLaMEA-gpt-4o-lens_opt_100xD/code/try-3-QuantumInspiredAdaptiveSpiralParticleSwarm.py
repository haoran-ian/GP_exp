import numpy as np

class QuantumInspiredAdaptiveSpiralParticleSwarm:
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
        angle_increment = 0.1
        initial_radius = 0.1
        radius = initial_radius

        def quantum_rotation_gate(pos, gbest, angle):
            theta = np.random.rand() * 2 * np.pi
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            quantum_displacement = np.dot(rotation_matrix, (gbest[:2] - pos[:2])) * angle
            return np.clip(pos[:2] + quantum_displacement, func.bounds.lb[:2], func.bounds.ub[:2])

        while self.evaluations < self.budget:
            for idx in range(num_particles):
                # Update velocity and position using PSO formula
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia = 0.7
                cognitive = 2.0 * r1 * (personal_best_positions[idx] - particles[idx])
                social = 1.5 * r2 * (global_best_position - particles[idx])
                velocities[idx] = inertia * velocities[idx] + cognitive + social
                particles[idx] += velocities[idx]
                particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                # Quantum-inspired spiral exploration
                if idx == global_best_index:
                    angle = self.evaluations * angle_increment
                    particles[idx][:2] = quantum_rotation_gate(particles[idx], global_best_position, angle)

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
                    radius = max(initial_radius, radius * 0.8)  # Adaptive radius reduction

                # Introduce mutation-based perturbation if no improvement
                if self.evaluations % 20 == 0:
                    particles[idx] += np.random.normal(0, 0.01, self.dim)
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                if self.evaluations >= self.budget:
                    break
            
            # Increase spiral search radius if no improvement
            if all(personal_best_values >= global_best_value):
                radius *= 1.2

        return global_best_position, global_best_value