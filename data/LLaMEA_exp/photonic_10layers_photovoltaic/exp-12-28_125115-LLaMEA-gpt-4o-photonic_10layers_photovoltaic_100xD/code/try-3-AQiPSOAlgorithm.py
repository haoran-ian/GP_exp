import numpy as np

class AQiPSOAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Number of particles
        self.inertia_weight = 0.9  # Higher initial inertia
        self.cognitive_coeff = 2.0  # Increased cognitive component
        self.social_coeff = 2.0  # Increased social component
        self.inertia_decay = 0.99  # Adaptive inertia weight decay
        self.quantum_coeff_base = 0.05  # Base quantum-inspired exploration coefficient
        self.quantum_coeff_adjust = 0.1  # Adaptive quantum coefficient for better exploration

    def __call__(self, func):
        # Initialize particle positions and velocities
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(x) for x in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size
        quantum_coeff = self.quantum_coeff_base

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive quantum-inspired exploration
                quantum_exploration = np.random.uniform(-1, 1, self.dim) * quantum_coeff
                # Update velocities with quantum exploration
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_coeff * r2 * (global_best_position - particles[i]) +
                                 quantum_exploration)
                particles[i] += velocities[i]

                # Ensure particles are within bounds
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate new positions
                score = func(particles[i])
                evaluations += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]
                    # Increase exploration when finding a new global best
                    quantum_coeff = min(quantum_coeff + self.quantum_coeff_adjust, 1.0)

            # Dynamically adjust parameters
            self.inertia_weight *= self.inertia_decay
            quantum_coeff = max(self.quantum_coeff_base, quantum_coeff * 0.95)  # Reduce exploration over time

        return global_best_position, global_best_score