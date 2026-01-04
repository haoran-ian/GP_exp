import numpy as np

class QiPSOAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Number of particles
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.learning_rate_decay = 0.99  # Adaptive learning rate parameter
        self.quantum_coeff = 0.05  # Quantum-inspired exploration coefficient

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

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum-inspired exploration
                quantum_exploration = np.random.uniform(-1, 1, self.dim) * self.quantum_coeff
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

            # Adaptively tune learning rates
            self.cognitive_coeff *= self.learning_rate_decay
            self.social_coeff *= self.learning_rate_decay

        return global_best_position, global_best_score